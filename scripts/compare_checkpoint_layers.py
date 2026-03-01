#!/usr/bin/env python3
"""
Standalone script: compute per-layer parameter differences between two checkpoints.

Outputs per-layer and total: L2/L1 diff, ratio vs ref, and sparsity_abs.
Sparsity (bf16): fraction of elements where (checkpoint - hf_model) == 0 in bfloat16.

Usage:
  python scripts/compare_checkpoint_layers.py --checkpoint /opt/tiger/verl/checkpoints/on-policy-distillation/gsm8k-math_qwen3-1.7b_grpo_rollout-8/global_step_1160/actor --hf_model Qwen/Qwen3-1.7B --output diff.json \\
    [--only-params]

Checkpoint can be:
  - A directory: HF-style (model.safetensors / pytorch_model.bin / *.safetensors), or
    verl FSDP actor dir (model_world_size_*_rank_*.pt + fsdp_config.json)
  - A .pt/.pth file: PyTorch state_dict (single dict of tensors)
HF model can be a HuggingFace model name or path to a local directory.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch


def _param_name_to_layer_name(param_name: str) -> str:
    """Extract layer name for grouping (e.g. model.layers.0.xxx -> layer_0)."""
    match = re.search(r"\.layers\.(\d+)\.", param_name)
    if match:
        return f"layer_{match.group(1)}"
    if "embed" in param_name or ("norm" in param_name and "layers" not in param_name):
        return "embed_or_norm"
    return "other"


def _normalize_key(key: str) -> str:
    """Strip common prefixes so keys from different save formats can match."""
    prefixes = ("_fsdp_wrapped_module.", "model.")
    while True:
        before = key
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key == before:
            break
    return key


def _load_fsdp_actor_checkpoint(path: Path, device: str = "cpu") -> dict:
    """Load and merge FSDP sharded checkpoint (verl actor dir: model_world_size_*_rank_*.pt)."""
    try:
        from torch.distributed._tensor import DTensor
    except ImportError:
        from torch.distributed.tensor import DTensor
    config_file = path / "fsdp_config.json"
    if not config_file.exists():
        raise FileNotFoundError("fsdp_config.json not found")
    with open(config_file) as f:
        config = json.load(f)
    world_size = config.get("world_size")
    if world_size is None:
        raise ValueError("world_size not in fsdp_config.json")
    model_files = [
        path / f"model_world_size_{world_size}_rank_{r}.pt"
        for r in range(world_size)
    ]
    for f in model_files:
        if not f.exists():
            raise FileNotFoundError(f"Expected {f}")
    # Load all shards (weights_only=False for DTensor)
    shards = []
    for r in range(world_size):
        shards.append(torch.load(str(model_files[r]), map_location=device, weights_only=False))
    # Merge: same keys in all shards; values are DTensor with _local_tensor
    state_dict = {}
    param_placements = {}
    keys = list(shards[0].keys())
    for key in keys:
        state_dict[key] = []
        for rank_sd in shards:
            t = rank_sd[key]
            if isinstance(t, DTensor):
                state_dict[key].append(t._local_tensor.float())
                placements = tuple(t.placements)
                if key not in param_placements:
                    param_placements[key] = placements
            else:
                state_dict[key].append(t.float() if t.is_floating_point() else t)
    for key in keys:
        if key in param_placements:
            placements = param_placements[key]
            # Find Shard placement (may be (Replicate, Shard) for ddp+fsdp)
            shard_dim = 0
            for p in placements:
                if hasattr(p, "dim"):
                    shard_dim = p.dim
                    break
            state_dict[key] = torch.cat(state_dict[key], dim=shard_dim).contiguous()
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0).contiguous()
    del shards
    return state_dict


def load_state_dict_from_path(path: str, device: str = "cpu"):
    """Load state dict from checkpoint path (dir with safetensors/bin or .pt file)."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        if path.suffix in (".pt", ".pth", ".bin"):
            state = torch.load(path, map_location=device, weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                return state["state_dict"]
            if isinstance(state, dict):
                return state
            raise ValueError(f"Unexpected checkpoint content type: {type(state)}")
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    # Directory: check for verl FSDP actor format first
    if (path / "fsdp_config.json").exists():
        model_pt = path / f"model_world_size_8_rank_0.pt"
        if not model_pt.exists():
            with open(path / "fsdp_config.json") as f:
                ws = json.load(f).get("world_size")
            if ws is not None:
                model_pt = path / f"model_world_size_{ws}_rank_0.pt"
        if model_pt.exists():
            print("Detected FSDP actor checkpoint, merging shards...", flush=True)
            return _load_fsdp_actor_checkpoint(path, device=device)

    # Directory: HF-style (safetensors or pytorch)
    safetensors_files = list(path.glob("*.safetensors"))
    if safetensors_files:
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("safetensors is required. pip install safetensors")
        # Single file or sharded
        if len(safetensors_files) == 1:
            return load_file(str(safetensors_files[0]), device=device)
        index_file = path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            state = {}
            weight_map = index.get("weight_map", {})
            shards_to_load = sorted(set(weight_map.values()))
            for shard_name in shards_to_load:
                st_path = path / shard_name
                if not st_path.exists():
                    continue
                part = load_file(str(st_path), device=device)
                for k, v in part.items():
                    state[k] = v
            return state
        # Multiple shards without index: load all and merge
        state = {}
        for f in sorted(safetensors_files):
            part = load_file(str(f), device=device)
            for k, v in part.items():
                if k not in state:
                    state[k] = v
        return state

    bin_file = path / "pytorch_model.bin"
    if bin_file.exists():
        return torch.load(bin_file, map_location=device, weights_only=True)

    model_safetensors = path / "model.safetensors"
    if model_safetensors.exists():
        from safetensors.torch import load_file
        return load_file(str(model_safetensors), device=device)

    raise FileNotFoundError(
        f"No model weights found under {path}. "
        "Expect model.safetensors, pytorch_model.bin, or *.safetensors shards."
    )


def load_state_dict_from_hf(model_name_or_path: str, device: str = "cpu"):
    """Load state dict from HuggingFace model (name or local path)."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError("transformers is required. pip install transformers")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    state = {k: v.to(device) for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return state


def align_state_dicts(state_a: dict, state_b: dict, normalize_keys: bool = True):
    """
    Return aligned (name_a, tensor_a, tensor_b) for names that exist in both with same shape.
    If normalize_keys, try to match by normalized names (strip common prefixes).
    """
    def to_key_map(sd):
        if not normalize_keys:
            return {k: k for k in sd}
        return {_normalize_key(k): k for k in sd}

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
    state_hf: dict,
    only_parameters: bool = False,
    normalize_keys: bool = True,
):
    """
    Compute per-layer L2/L1 diff and ratio between checkpoint and HF model.
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
        state_checkpoint, state_hf, normalize_keys=normalize_keys
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

        layer_name = _param_name_to_layer_name(name)
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
            "No matching parameter keys between checkpoint and HF model. "
            "Try --no-normalize-keys if keys use the same naming, or check that both are the same architecture."
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-layer parameter differences between trained checkpoint and HF model."
    )
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to trained checkpoint (dir or .pt file)")
    parser.add_argument("--hf_model", "-m", required=True, help="HuggingFace model name or path (e.g. Qwen/Qwen3-1.7B)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path (default: print to stdout)")
    parser.add_argument("--only-params", action="store_true", help="Skip known buffer keys (e.g. rotary)")
    parser.add_argument("--no-normalize-keys", action="store_true", help="Do not try to match keys by stripping prefixes")
    parser.add_argument("--device", default="cpu", help="Device for tensor ops (default: cpu)")
    args = parser.parse_args()

    device = args.device
    print("Loading checkpoint state...", flush=True)
    state_ckpt = load_state_dict_from_path(args.checkpoint, device=device)
    print(f"  Keys: {len(state_ckpt)}", flush=True)

    print("Loading HF model state...", flush=True)
    state_hf = load_state_dict_from_hf(args.hf_model, device=device)
    print(f"  Keys: {len(state_hf)}", flush=True)

    print("Computing per-layer diffs...", flush=True)
    result = compute_layer_diffs(
        state_ckpt,
        state_hf,
        only_parameters=args.only_params,
        normalize_keys=not args.no_normalize_keys,
    )
    # Make result JSON-serializable (remove tensor values in param_diffs if any)
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, torch.Tensor):
            return obj.tolist() if obj.numel() < 10 else f"<tensor shape={tuple(obj.shape)}>"
        return str(obj)

    result_ser = to_serializable(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_ser, f, indent=2, ensure_ascii=False)
        print(f"Saved to {out_path}", flush=True)

    # Print summary
    total_num = result_ser.get("total_param_num", 0)
    print(f"\nTotal params compared: {total_num:,}")
    print(f"Total diff L2: {result_ser.get('total_layer_l2', 0):.6f}  ref L2: {result_ser.get('total_ref_l2', 0):.6f}  ratio_l2: {result_ser.get('total_ratio_l2')}")
    print(f"Total diff L1: {result_ser.get('total_layer_l1', 0):.6f}  ref L1: {result_ser.get('total_ref_l1', 0):.6f}  ratio_l1: {result_ser.get('total_ratio_l1')}")
    print(f"Sparsity (bf16, diff==0): total_unchange={result_ser.get('total_unchange_param_abs_num', 0):,}  total_sparsity_abs={result_ser.get('total_sparsity_abs', 0):.4f}")
    print("\nPer-layer summary:")
    for layer_name in sorted(result_ser.keys()):
        if layer_name.startswith("total_"):
            continue
        layer_data = result_ser.get(layer_name)
        if not isinstance(layer_data, dict) or "layer_l2" not in layer_data:
            continue
        r2 = layer_data.get("layer_ratio_l2")
        r2str = f"{r2:.4f}" if r2 is not None else "N/A"
        sp_abs = layer_data.get("layer_sparsity_abs")
        sp_str = f" sparsity_abs={sp_abs:.4f}" if sp_abs is not None else ""
        print(
            f"  {layer_name}: L2={layer_data['layer_l2']:.6f} ref_L2={layer_data['layer_ref_l2']:.6f} "
            f"ratio_l2={r2str} numel={layer_data.get('layer_total_param_num', 0):,}{sp_str}"
        )


if __name__ == "__main__":
    main()
