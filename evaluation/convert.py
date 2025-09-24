import os, torch, json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HF_DIR = "/media/volume/MultiAgent-Data/checkpoints/msra-sft/eswp_prune_.2_mb_8_H200_lr_1e-5_numina_30k_qwen2.5-math-1.5b/global_step_2000/huggingface"
PT     = "/media/volume/MultiAgent-Data/checkpoints/msra-sft/eswp_prune_.2_mb_8_H200_lr_1e-5_numina_30k_qwen2.5-math-1.5b/global_step_2000/model_world_size_1_rank_0.pt"
OUT    = "/media/volume/MultiAgent-Data/checkpoints/msra-sft/eswp_prune_.2_mb_8_H200_lr_1e-5_numina_30k_qwen2.5-math-1.5b/global_step_2000/full_model"

def convert_dtensor_to_tensor(state_dict):
    """将DTensor转换为普通tensor"""
    clean_state_dict = {}
    
    for key, value in state_dict.items():
        if hasattr(value, 'to_local'):
            # 如果是DTensor，转换为本地tensor
            try:
                clean_value = value.to_local()
                print(f"Converted DTensor {key}: {value.shape} -> {clean_value.shape}")
            except Exception as e:
                print(f"Warning: Failed to convert {key}: {e}")
                # 尝试其他方法
                if hasattr(value, '_local_tensor'):
                    clean_value = value._local_tensor
                elif hasattr(value, 'full_tensor'):
                    clean_value = value.full_tensor()
                else:
                    clean_value = value
        elif torch.is_tensor(value):
            # 如果已经是普通tensor，直接使用
            clean_value = value
        else:
            # 其他类型，尝试转换
            try:
                clean_value = torch.tensor(value) if not torch.is_tensor(value) else value
            except:
                clean_value = value
                
        clean_state_dict[key] = clean_value
    
    return clean_state_dict

# 1) 读取已有的 config 与 tokenizer
config = AutoConfig.from_pretrained(HF_DIR, trust_remote_code=True)
tok    = AutoTokenizer.from_pretrained(HF_DIR, use_fast=True, trust_remote_code=True)
print("Loaded config and tokenizer")

# 2) 按 config 构建"同骨架"空模型
model  = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
model  = model.eval()
print("Created model from config")

# 3) 读取并清洗 state_dict
print(f"Loading checkpoint from {PT}")
state = torch.load(PT, weights_only=False, map_location='cpu')

# 获取state_dict
if "state_dict" in state:
    sd = state["state_dict"]
elif "model_state_dict" in state:
    sd = state["model_state_dict"]
else:
    sd = state

print(f"Found {len(sd)} keys in checkpoint")

# 转换DTensor到普通tensor
print("Converting DTensors to regular tensors...")
sd = convert_dtensor_to_tensor(sd)

# 清理键名
clean = {}
for k, v in sd.items():
    # 移除各种前缀
    original_k = k
    k = (k.replace("module.", "")
         .replace("_fsdp_wrapped_module.", "")
         .replace("_forward_module.", "")
         .replace("_orig_mod.", ""))
    
    if k != original_k:
        print(f"Key renamed: {original_k} -> {k}")
    
    clean[k] = v

print(f"Cleaned state_dict has {len(clean)} keys")

# 检查模型和checkpoint的键
model_keys = set(model.state_dict().keys())
checkpoint_keys = set(clean.keys())

print(f"Model has {len(model_keys)} parameters")
print(f"Checkpoint has {len(checkpoint_keys)} parameters")

missing_in_checkpoint = model_keys - checkpoint_keys
unexpected_in_checkpoint = checkpoint_keys - model_keys

if missing_in_checkpoint:
    print(f"Missing in checkpoint: {list(missing_in_checkpoint)[:5]}")
if unexpected_in_checkpoint:
    print(f"Unexpected in checkpoint: {list(unexpected_in_checkpoint)[:5]}")

# 4) 加载state_dict
try:
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"Successfully loaded state_dict")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    if missing:
        print(f"Missing keys sample: {missing[:3]}")
    if unexpected:
        print(f"Unexpected keys sample: {unexpected[:3]}")
        
except Exception as e:
    print(f"Error loading state_dict: {e}")
    raise

# 5) 导出为 HF 目录
os.makedirs(OUT, exist_ok=True)
print(f"Saving model to {OUT}")

try:
    model.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    print("✅ Successfully saved HF model to:", OUT)
except Exception as e:
    print(f"❌ Error saving model: {e}")
    # 尝试不使用safe_serialization
    try:
        model.save_pretrained(OUT, safe_serialization=False)
        tok.save_pretrained(OUT)
        print("✅ Successfully saved HF model (without safe_serialization) to:", OUT)
    except Exception as e2:
        print(f"❌ Failed to save model: {e2}")
        raise