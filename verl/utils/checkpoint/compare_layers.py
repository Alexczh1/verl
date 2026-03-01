# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Per-layer parameter difference between two state dicts (e.g. current checkpoint vs initial model).
Used by scripts/compare_checkpoint_layers.py and by FSDP SFT trainer during validation.
"""

import re
from collections import defaultdict

import torch


def param_name_to_layer_name(param_name: str) -> str:
    """Extract layer name for grouping (e.g. model.layers.0.xxx -> layer_0)."""
    match = re.search(r"\.layers\.(\d+)\.", param_name)
    if match:
        return f"layer_{match.group(1)}"
    if "embed" in param_name or ("norm" in param_name and "layers" not in param_name):
        return "embed_or_norm"
    return "other"


def normalize_key(key: str) -> str:
    """Strip common prefixes so keys from different save formats can match."""
    prefixes = ("_fsdp_wrapped_module.", "model.")
    while True:
        before = key
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key == before:
            break
    return key


def align_state_dicts(state_a: dict, state_b: dict, normalize_keys: bool = True):
    """
    Return aligned (name_a, tensor_a, tensor_b) for names that exist in both with same shape.
    If normalize_keys, try to match by normalized names (strip common prefixes).
    """
    def to_key_map(sd):
        if not normalize_keys:
            return {k: k for k in sd}
        return {normalize_key(k): k for k in sd}

    keys_a = to_key_map(state_a)
    keys_b = to_key_map(state_b)
    common_norm = set(keys_a) & set(keys_b)
    for norm in common_norm:
        ka, kb = keys_a[norm], keys_b[norm]
        ta = state_a[ka]
        tb = state_b[kb]
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            continue
        if ta.shape != tb.shape:
            continue
        yield (ka, ta, tb)


def compute_layer_diffs(
    state_checkpoint: dict,
    state_ref: dict,
    only_parameters: bool = False,
    normalize_keys: bool = True,
):
    """
    Compute per-layer L2/L1 diff and ratio between checkpoint and reference (e.g. initial) state.
    Returns dict with layer_name -> stats and totals.
    """
    layer_diffs = defaultdict(lambda: {
        "layer_l2_sq": 0.0,
        "layer_l1": 0.0,
        "layer_ref_l2_sq": 0.0,
        "layer_ref_l1": 0.0,
        "layer_total_param_num": 0,
        "layer_unchange_param_abs_num": 0,
        "param_diffs": {},
    })

    total_l2_sq = 0.0
    total_l1 = 0.0
    total_ref_l2_sq = 0.0
    total_ref_l1 = 0.0
    total_num = 0
    total_unchange_abs = 0
    num_matched = 0

    for name, p_ckpt, p_ref in align_state_dicts(
        state_checkpoint, state_ref, normalize_keys=normalize_keys
    ):
        if only_parameters:
            if "inv_freq" in name or "rotary" in name.lower() or "sin_cos" in name.lower():
                continue
        p_ckpt = p_ckpt.float()
        p_ref = p_ref.float()
        diff = p_ckpt - p_ref
        total_param_num = diff.numel()
        # Sparsity (bf16): count elements where diff == 0 in bfloat16 sense
        diff_bf16 = diff.to(torch.bfloat16)
        unchange_param_abs_num = (diff_bf16 == 0).sum().item()
        sparsity_abs = unchange_param_abs_num / total_param_num if total_param_num > 0 else 0.0

        diff_l2_sq = (diff * diff).sum().item()
        diff_l1 = diff.abs().sum().item()
        ref_l2_sq = (p_ref * p_ref).sum().item()
        ref_l1 = p_ref.abs().sum().item()
        ref_l2 = ref_l2_sq ** 0.5
        ratio_l2 = (diff_l2_sq ** 0.5) / ref_l2 if ref_l2 > 0 else None
        ratio_l1 = diff_l1 / ref_l1 if ref_l1 > 0 else None

        layer_name = param_name_to_layer_name(name)
        layer_diffs[layer_name]["layer_l2_sq"] += diff_l2_sq
        layer_diffs[layer_name]["layer_l1"] += diff_l1
        layer_diffs[layer_name]["layer_ref_l2_sq"] += ref_l2_sq
        layer_diffs[layer_name]["layer_ref_l1"] += ref_l1
        layer_diffs[layer_name]["layer_total_param_num"] += total_param_num
        layer_diffs[layer_name]["layer_unchange_param_abs_num"] += unchange_param_abs_num
        layer_diffs[layer_name]["param_diffs"][name] = {
            "l2": (diff_l2_sq ** 0.5),
            "l1": diff_l1,
            "ref_l2": ref_l2,
            "ref_l1": ref_l1,
            "ratio_l2": ratio_l2,
            "ratio_l1": ratio_l1,
            "numel": total_param_num,
            "unchange_param_abs_num": unchange_param_abs_num,
            "sparsity_abs": sparsity_abs,
        }

        total_l2_sq += diff_l2_sq
        total_l1 += diff_l1
        total_ref_l2_sq += ref_l2_sq
        total_ref_l1 += ref_l1
        total_num += total_param_num
        total_unchange_abs += unchange_param_abs_num
        num_matched += 1

    if num_matched == 0:
        raise ValueError(
            "No matching parameter keys between checkpoint and reference state. "
            "Try normalize_keys=False if keys use the same naming, or check that both are the same architecture."
        )

    # Build result with layer-level and totals
    result = {
        "total_layer_l2": total_l2_sq ** 0.5,
        "total_layer_l1": total_l1,
        "total_ref_l2": total_ref_l2_sq ** 0.5,
        "total_ref_l1": total_ref_l1,
        "total_param_num": total_num,
        "total_ratio_l2": (total_l2_sq ** 0.5) / (total_ref_l2_sq ** 0.5) if total_ref_l2_sq > 0 else None,
        "total_ratio_l1": total_l1 / total_ref_l1 if total_ref_l1 > 0 else None,
        "total_unchange_param_abs_num": total_unchange_abs,
        "total_sparsity_abs": total_unchange_abs / total_num if total_num > 0 else 0.0,
    }
    for layer_name, data in sorted(layer_diffs.items(), key=lambda x: x[0]):
        layer_l2 = data["layer_l2_sq"] ** 0.5
        layer_ref_l2 = data["layer_ref_l2_sq"] ** 0.5 if data["layer_ref_l2_sq"] > 0 else 0.0
        layer_total = data["layer_total_param_num"]
        layer_unchange_abs = data["layer_unchange_param_abs_num"]
        result[layer_name] = {
            "layer_l2": layer_l2,
            "layer_l1": data["layer_l1"],
            "layer_ref_l2": layer_ref_l2,
            "layer_ref_l1": data["layer_ref_l1"],
            "layer_ratio_l2": layer_l2 / layer_ref_l2 if layer_ref_l2 > 0 else None,
            "layer_ratio_l1": data["layer_l1"] / data["layer_ref_l1"] if data["layer_ref_l1"] > 0 else None,
            "layer_total_param_num": layer_total,
            "layer_unchange_param_abs_num": layer_unchange_abs,
            "layer_sparsity_abs": layer_unchange_abs / layer_total if layer_total > 0 else 0.0,
            "param_diffs": data["param_diffs"],
        }
    return result
