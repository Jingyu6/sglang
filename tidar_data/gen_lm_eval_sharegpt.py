# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import inspect
import json
import os
import random
import sys
from typing import Any, Dict, Iterable, Optional

from tqdm import tqdm


def _import_lm_eval():
    try:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        from lm_eval import tasks  # type: ignore
    except Exception as e:
        print(
            "ERROR: Could not import lm_eval (EleutherAI/lm-evaluation-harness). "
            "Please install it, e.g.: pip install lm-eval==0.4.*",
            file=sys.stderr,
        )
        raise
    return tasks

def _resolve_task(tasks_mod, task_name: str):
    return tasks_mod.get_task_dict(task_name)[task_name]

def _iter_docs(task, split: str) -> Iterable[Dict[str, Any]]:
    # Try common split access patterns across harness versions
    if split == "validation":
        if hasattr(task, "validation_docs"):
            docs = task.validation_docs()
        elif hasattr(task, "val_docs"):
            docs = task.val_docs()
        else:
            raise ValueError("Task does not expose validation docs")
    elif split == "test":
        if hasattr(task, "test_docs"):
            docs = task.test_docs()
        else:
            raise ValueError("Task does not expose test docs")
    elif split == "train":
        if hasattr(task, "training_docs"):
            docs = task.training_docs()
        else:
            raise ValueError("Task does not expose training docs")
    else:
        raise ValueError(f"Unknown split: {split}")
    return docs

def _fewshot_context(task, doc: Dict[str, Any], num_shots: int, seed: int) -> str:
    # Build kwargs based on the actual signature to avoid unexpected kw errors across versions
    params = inspect.signature(task.fewshot_context).parameters
    kwargs: Dict[str, Any] = {"doc": doc, "num_fewshot": num_shots}

    # Prefer numpy RandomState when available; fall back to random.Random
    rng_val: Any
    try:
        import numpy as _np  # type: ignore
        rng_val = _np.random.RandomState(seed)
    except Exception:
        rng_val = random.Random(seed)

    # Only pass parameters that exist
    if "include_target" in params:
        kwargs["include_target"] = False
    # rng vs rnd vs fewshot_random_seed
    if "rng" in params:
        kwargs["rng"] = rng_val
    elif "rnd" in params:
        kwargs["rnd"] = rng_val
    if "fewshot_random_seed" in params:
        kwargs["fewshot_random_seed"] = seed

    try:
        return task.fewshot_context(**kwargs)  # type: ignore[arg-type]
    except Exception as e:
        available = list(params.keys())
        raise RuntimeError(
            f"Failed to build few-shot context with supported kwargs. "
            f"Available params: {available}. Error: {e}"
        )

def _get_task_config_dict(task) -> Dict[str, Any]:
    cfg = getattr(task, "config", None)
    if cfg is None:
        cfg = getattr(task, "_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    # Some configs are pydantic/simple objects with to_dict()
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    # Best-effort attribute dict
    try:
        return dict(cfg)
    except Exception:
        pass
    return {}

def _split_kind_from_config(task) -> str:
    cfg = _get_task_config_dict(task)
    return cfg['test_split']

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def _generate_items_for_task(tasks_mod, task_name: str, num_shots: int, limit: Optional[int], seed: int):
    task = _resolve_task(tasks_mod, task_name)
    if hasattr(task, "download"):
        task.download()
    split_to_use = _split_kind_from_config(task)
    docs = _iter_docs(task, split_to_use)

    # Configure fewshot exactly like harness
    try:
        if hasattr(task, "set_config"):
            task.set_config("num_fewshot", num_shots)
        else:
            cfg = getattr(task, "config", None) or getattr(task, "_config", None)
            if isinstance(cfg, dict):
                cfg["num_fewshot"] = num_shots
            else:
                try:
                    setattr(task.config, "num_fewshot", num_shots)
                except Exception:
                    pass
        if hasattr(task, "set_fewshot_seed"):
            task.set_fewshot_seed(seed)
    except Exception:
        pass

    items = []
    for idx, doc in tqdm(
        enumerate(docs), 
        total=len(docs), 
        desc=f"Generating items for task {task_name}"
    ):
        if idx == limit:
            break
        prompt = _fewshot_context(task, doc, num_shots=num_shots, seed=seed)
        items.append(
            {
                "id": f"{task_name}-{idx}",
                "task": task_name,
                "num_shots": num_shots,
                "seed": seed,
                "conversations": [{"from": "human", "value": prompt}],
            }
        )
    return items

def main():
    parser = argparse.ArgumentParser(
        description="Dump lm-eval-harness prompts (few-shot contexts) in ShareGPT jsonl format without running any model."
    )
    parser.add_argument("--tasks", type=str, help="Comma-separated list of task names.", required=True)
    parser.add_argument("--output-path", type=str, help="Output jsonl path.", required=True)
    parser.add_argument("--num-shots", type=str, default=None, help="Number of few-shot examples (n-shot).")
    parser.add_argument("--limits", type=str, default=None, help="Max number of prompts to dump.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for few-shot sampling.")
    args = parser.parse_args()

    # Aggregate task names
    task_names = []
    num_shots_list = []
    limits_list = []
    task_names.extend([t.strip() for t in args.tasks.split(",") if t.strip()])
    if args.num_shots:
        num_shots_list.extend([int(s.strip()) for s in args.num_shots.split(",") if s.strip()])
    else:
        num_shots_list = [0] * len(task_names)
    if args.limits:
        limits_list.extend([int(l.strip()) for l in args.limits.split(",") if l.strip()])
    else:
        limits_list = [None] * len(task_names)   

    # Decide default output path
    tasks_mod = _import_lm_eval()
    all_items = []
    for name, num_shots, limit in zip(task_names, num_shots_list, limits_list):
        print(f"Generating samples from task {name} with {num_shots} shots and {limit} limit...")
        all_items.extend(
            _generate_items_for_task(
                tasks_mod, name, num_shots=num_shots, limit=limit, seed=args.seed
            )
        )

    _ensure_parent_dir(args.output_path)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(all_items)} ShareGPT prompts from {len(task_names)} task(s) to {args.output_path}")


if __name__ == "__main__":
    main()
