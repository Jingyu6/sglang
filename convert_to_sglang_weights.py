import argparse
import json
import os
import shutil
from collections import defaultdict

import torch
from safetensors.torch import save_file


def load_checkpoint_state_dict(ckpt_path: str) -> dict:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "module", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    raise ValueError(f"Unsupported checkpoint object in {ckpt_path}")


def save_sharded_safetensors(state_dict: dict, index_path: str, out_dir: str):
    with open(index_path, "r") as f:
        index_json = json.load(f)
    weight_map: dict = index_json["weight_map"]

    shard_to_params = defaultdict(dict)
    missing = []
    for param_name, shard_file in weight_map.items():
        if param_name not in state_dict:
            missing.append(param_name)
            continue
        shard_to_params[shard_file][param_name] = state_dict[param_name].contiguous()

    if missing:
        raise KeyError(
            f"Missing {len(missing)} params in checkpoint that exist in index.json, e.g. {missing[:5]}"
        )

    os.makedirs(out_dir, exist_ok=True)
    for shard_file, shard_tensors in shard_to_params.items():
        shard_out = os.path.join(out_dir, shard_file)
        os.makedirs(os.path.dirname(shard_out), exist_ok=True)
        save_file(shard_tensors, shard_out)

    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_json, f, indent=2)


def write_tidar_config(src_config_path: str, dst_config_path: str):
    with open(src_config_path, "r") as f:
        cfg = json.load(f)
    cfg["architectures"] = ["TiDARForCausalLM"]
    with open(dst_config_path, "w") as f:
        json.dump(cfg, f, indent=2)


def copy_non_weight_files(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    skip_names = {"LICENSE", "README.md", "model.safetensors", "model.safetensors.index.json"}
    for root, _, files in os.walk(src_dir):
        rel_root = os.path.relpath(root, src_dir)
        for name in files:
            if name in skip_names or name.endswith(".safetensors"):
                continue
            src = os.path.join(root, name)
            rel = os.path.normpath(os.path.join(rel_root, name)) if rel_root != "." else name
            dst = os.path.join(dst_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def main():
    """
    python convert_to_sglang_weights.py \
        --src-model-dir /lustre/fsw/portfolios/nvr/users/jinliu/public_models/Qwen3-8B \
        --checkpoint /lustre/fsw/portfolios/nvr/users/jinliu/megatron_exp/tidar_8b.pt \
        --out-dir /lustre/fsw/portfolios/nvr/users/jinliu/megatron_exp/tidar_8b_sglang
    """
    parser = argparse.ArgumentParser(
        description="Save checkpoint as sharded safetensors matching original layout, update config to TiDARForCausalLM, and copy non-weight files."
    )
    parser.add_argument("--src-model-dir", required=True, help="Original HF model directory")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path (.pt/.bin)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    src_model_dir = args.src_model_dir
    ckpt_path = args.checkpoint
    out_dir = args.out_dir

    index_path = os.path.join(src_model_dir, "model.safetensors.index.json")
    config_path = os.path.join(src_model_dir, "config.json")

    print("[TiDAR] Loading checkpoint ...")
    state_dict = load_checkpoint_state_dict(ckpt_path)

    if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
        print("[TiDAR] Tying lm_head.weight to model.embed_tokens.weight")
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    print(f"[TiDAR] Saving sharded weights to {out_dir}")
    save_sharded_safetensors(state_dict, index_path, out_dir)

    print("[TiDAR] Copying non-weight files ...")
    copy_non_weight_files(src_model_dir, out_dir)

    print("[TiDAR] Writing TiDAR config.json ...")
    write_tidar_config(config_path, os.path.join(out_dir, "config.json"))

    print("[TiDAR] Done.")


if __name__ == "__main__":
    main()


