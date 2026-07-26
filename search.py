# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
from PIL import Image

import models.kimi as kimi


# ============================================================
# dist utils
# ============================================================

def dist_is_initialized() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def get_rank() -> int:
    if dist_is_initialized():
        import torch.distributed as dist
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    if dist_is_initialized():
        import torch.distributed as dist
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> Tuple[int, int, int]:
    import torch
    import torch.distributed as dist

    local_rank = get_local_rank()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def barrier() -> None:
    if dist_is_initialized():
        import torch.distributed as dist
        dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not dist_is_initialized():
        return obj
    import torch.distributed as dist
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def all_gather_objects(obj: Any) -> List[Any]:
    if not dist_is_initialized():
        return [obj]
    import torch.distributed as dist
    world_size = dist.get_world_size()
    out = [None for _ in range(world_size)]
    dist.all_gather_object(out, obj)
    return out


def all_reduce_sum_scalar(value: float, device) -> float:
    import torch
    import torch.distributed as dist

    if not dist_is_initialized():
        return float(value)
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def all_reduce_sum_vector(values: Sequence[float], device) -> List[float]:
    import torch
    import torch.distributed as dist

    if not dist_is_initialized():
        return [float(v) for v in values]
    if len(values) == 0:
        return []
    t = torch.tensor(list(values), dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(x) for x in t.tolist()]


# ============================================================
# utils
# ============================================================

def log_print(*args, main_only: bool = True, **kwargs) -> None:
    if (not main_only) or is_main_process():
        print(*args, **kwargs, flush=True)



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass



def scalarize(x: Any) -> Any:
    if isinstance(x, pd.Series):
        if len(x) == 1:
            return scalarize(x.iloc[0])
        return x.to_dict()
    if isinstance(x, np.generic):
        return x.item()
    return x


def format_elapsed_human(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    days, rem = divmod(total, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    parts.append(f"({seconds:.2f}s)")
    return " ".join(parts)


# ============================================================
# parquet / image
# ============================================================

def read_parquet_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()



def decode_image(value: Any) -> Image.Image:
    value = scalarize(value)

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, str):
        if os.path.exists(value):
            return Image.open(value).convert("RGB")
        raise FileNotFoundError(f"Image path does not exist: {value}")
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            return decode_image(value["bytes"])
        if "path" in value and value["path"] is not None:
            return decode_image(value["path"])
        if "array" in value and value["array"] is not None:
            arr = np.asarray(value["array"])
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
    if hasattr(value, "shape"):
        arr = np.asarray(value)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    raise TypeError(f"Unsupported image value type: {type(value)}")



def guess_question_col(inst_df: pd.DataFrame) -> str:
    for c in ["question", "instruction", "prompt", "text", "query"]:
        if c in inst_df.columns:
            return c
    raise ValueError(f"Could not infer question column from {list(inst_df.columns)}")



def guess_answer_col(inst_df: pd.DataFrame) -> str:
    for c in ["answer", "label", "target", "response"]:
        if c in inst_df.columns:
            return c
    raise ValueError(f"Could not infer answer column from {list(inst_df.columns)}")



def guess_image_col(img_df: pd.DataFrame) -> str:
    for c in ["image", "jpg", "png", "bytes", "image_bytes", "img", "image_path", "path"]:
        if c in img_df.columns:
            return c
    raise ValueError(f"Could not infer image column from {list(img_df.columns)}")



def build_records_from_two_parquets(
    inst_path: str,
    img_path: str,
    limit: int,
    inst_key_col: Optional[str] = None,
    img_key_col: Optional[str] = None,
    question_col: Optional[str] = None,
    answer_col: Optional[str] = None,
    image_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    inst_df = read_parquet_df(inst_path)
    img_df = read_parquet_df(img_path)

    question_col = question_col or guess_question_col(inst_df)
    answer_col = answer_col or guess_answer_col(inst_df)
    image_col = image_col or guess_image_col(img_df)

    inst_key_col = inst_key_col or ("imageId" if "imageId" in inst_df.columns else None)
    img_key_col = img_key_col or ("id" if "id" in img_df.columns else None)

    records: List[Dict[str, Any]] = []

    if inst_key_col is not None and img_key_col is not None:
        merged = inst_df.merge(
            img_df[[img_key_col, image_col]],
            left_on=inst_key_col,
            right_on=img_key_col,
            how="inner",
            suffixes=("_inst", "_img"),
        )
        if len(merged) == 0:
            raise RuntimeError(
                f"Merge produced 0 rows on left_key={inst_key_col}, right_key={img_key_col}"
            )

        for _, row in merged.iloc[:limit].iterrows():
            records.append({
                "id": scalarize(row.get("id_inst", row[inst_key_col])),
                "image_id": scalarize(row[inst_key_col]),
                "question": str(scalarize(row[question_col])),
                "answer": str(scalarize(row[answer_col])),
                "image": row[image_col],
            })
        return records

    n = min(len(inst_df), len(img_df), limit)
    for i in range(n):
        inst_row = inst_df.iloc[i]
        img_row = img_df.iloc[i]
        rec_id = scalarize(inst_row["id"]) if "id" in inst_df.columns else i
        records.append({
            "id": rec_id,
            "image_id": scalarize(inst_row["imageId"]) if "imageId" in inst_df.columns else i,
            "question": str(scalarize(inst_row[question_col])),
            "answer": str(scalarize(inst_row[answer_col])),
            "image": img_row[image_col],
        })
    return records



def shard_records(records: List[Dict[str, Any]], world_size: int, rank: int) -> List[Dict[str, Any]]:
    return records[rank::world_size]


# ============================================================
# prompt / inputs
# ============================================================

def build_messages(question: str):
    prompt = (
        "Answer the following question using a single word or short phrase only.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]



def prepare_single_input(processor, image: Image.Image, question: str):
    messages = build_messages(question)
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    return processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )


# ============================================================
# model / policy
# ============================================================


def apply_policy_to_model(model, moe_layer_indices: List[int], policy: Sequence[int]):
    """Dynamically apply policy to loaded model without reloading."""
    for local_i, layer_idx in enumerate(moe_layer_indices):
        layer = model.language_model.model.layers[layer_idx]
        new_gate_dict = {
            "text": 6,
            "visual": int(policy[local_i]),
        }
        layer.mlp.gate_dict = new_gate_dict
        layer.mlp.gate.gate_dict = new_gate_dict

def discover_moe_layer_indices(model) -> List[int]:
    config = model.config.text_config
    moe_layer_indices = []
    for idx, _layer in enumerate(model.language_model.model.layers):
        if (
            config.n_routed_experts is not None
            and idx >= config.first_k_dense_replace
            and idx % config.moe_layer_freq == 0
        ):
            moe_layer_indices.append(idx)
    return moe_layer_indices



def build_layer_gate_dict(
    model,
    policy: Sequence[int],
    freq_path: Optional[str] = None,
) -> Dict[Any, Any]:
    gate_dict: Dict[Any, Any] = {}
    if freq_path is not None:
        gate_dict["freq_path"] = freq_path

    moe_layer_indices = discover_moe_layer_indices(model)
    if len(policy) != len(moe_layer_indices):
        raise ValueError(f"Policy length {len(policy)} != #MoE layers {len(moe_layer_indices)}")

    for local_i, layer_idx in enumerate(moe_layer_indices):
        gate_dict[layer_idx] = {
            "text": 6,
            "visual": int(policy[local_i]),
        }
    return gate_dict



def load_model_with_policy(model_path: str, policy: Sequence[int], freq_path: Optional[str], device_str: str):
    import torch

    dummy_model, _ = kimi.load_model(model_path=model_path, layer_gate_dict=None, device_map="cpu")
    gate_dict = build_layer_gate_dict(dummy_model, policy, freq_path=freq_path)

    try:
        del dummy_model
        torch.cuda.empty_cache()
    except Exception:
        pass

    model, processor = kimi.load_model(
        model_path=model_path,
        layer_gate_dict=gate_dict,
        device_map=device_str,
    )
    return model, processor



def load_full_model(model_path: str, device_str: str):
    model, processor = kimi.load_model(
        model_path=model_path,
        layer_gate_dict=None,
        device_map=device_str,
    )
    return model, processor


# ============================================================
# token profile / budget
# ============================================================

def infer_token_profile(records, processor, media_placeholder_token_id: int, special_token_ids: Sequence[int]):
    visual_counts = []
    text_counts = []
    special_token_ids = set(int(x) for x in special_token_ids)

    for rec in records:
        image = decode_image(rec["image"])
        batch = prepare_single_input(processor, image, rec["question"])
        input_ids = batch["input_ids"][0].tolist()
        visual = sum(1 for x in input_ids if int(x) == int(media_placeholder_token_id))
        text = sum(1 for x in input_ids if int(x) not in special_token_ids and int(x) != int(media_placeholder_token_id))
        visual_counts.append(visual)
        text_counts.append(text)

    return float(np.mean(visual_counts)), float(np.mean(text_counts))



def compute_target_skip_units(
    target_skip_ratio: float,
    n_moe_layers: int,
    avg_visual_tokens: float,
    avg_text_tokens: float,
) -> int:
    if avg_visual_tokens <= 1e-12:
        return 0
    target_units = target_skip_ratio * 6.0 * n_moe_layers * (avg_visual_tokens + avg_text_tokens) / avg_visual_tokens
    target_units = int(round(target_units))
    target_units = max(0, min(6 * n_moe_layers, target_units))
    return target_units



def policy_skip_units(policy: Sequence[int]) -> int:
    return int(sum(6 - int(k) for k in policy))


# ============================================================
# KL helpers
# ============================================================

def log_softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    return x - np.log(np.sum(np.exp(x)) + 1e-12)



def topk_logprobs_from_logits(logits_1d: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    vocab = logits_1d.shape[0]
    k = min(topk, vocab)
    idx = np.argpartition(logits_1d, -k)[-k:]
    idx = idx[np.argsort(logits_1d[idx])[::-1]]
    lp = log_softmax_np(logits_1d)[idx]
    return idx.astype(np.int64), lp.astype(np.float64)



def sparse_topk_kl(
    p_ids: np.ndarray,
    p_logprobs: np.ndarray,
    q_ids: np.ndarray,
    q_logprobs: np.ndarray,
    floor_logprob: float = -30.0,
) -> float:
    union_ids = np.unique(np.concatenate([p_ids, q_ids], axis=0))
    p_map = {int(i): float(lp) for i, lp in zip(p_ids.tolist(), p_logprobs.tolist())}
    q_map = {int(i): float(lp) for i, lp in zip(q_ids.tolist(), q_logprobs.tolist())}

    p_vec = np.array([p_map.get(int(t), floor_logprob) for t in union_ids], dtype=np.float64)
    q_vec = np.array([q_map.get(int(t), floor_logprob) for t in union_ids], dtype=np.float64)

    p_vec = log_softmax_np(p_vec)
    q_vec = log_softmax_np(q_vec)

    p = np.exp(p_vec)
    kl = np.sum(p * (p_vec - q_vec))
    return float(kl)


# ============================================================
# baseline cache
# ============================================================

@dataclass
class BaselineSample:
    sample_id: str
    question: str
    answer: str
    generated_ids: List[int]
    step_top_ids: List[np.ndarray]
    step_top_logprobs: List[np.ndarray]



def build_baseline_cache(
    model,
    processor,
    records: List[Dict[str, Any]],
    max_new_tokens: int,
    baseline_topk: int,
) -> List[BaselineSample]:
    import torch

    # Model already loaded and in eval mode

    cache: List[BaselineSample] = []

    for i, rec in enumerate(records):
        image = decode_image(rec["image"])
        batch = prepare_single_input(processor, image, rec["question"])
        batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}

        with torch.no_grad():
            gen = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=False,
            )

        input_len = int(batch["input_ids"].shape[1])
        gen_ids = gen[0, input_len:].tolist()

        if len(gen_ids) == 0:
            cache.append(BaselineSample(
                sample_id=str(rec["id"]),
                question=str(rec["question"]),
                answer=str(rec["answer"]),
                generated_ids=[],
                step_top_ids=[],
                step_top_logprobs=[],
            ))
            continue

        full_ids = torch.tensor(
            [batch["input_ids"][0].tolist() + gen_ids],
            dtype=batch["input_ids"].dtype,
            device=model.device,
        )

        forward_kwargs = dict(batch)
        forward_kwargs["input_ids"] = full_ids
        if "attention_mask" in forward_kwargs:
            old_attn = forward_kwargs["attention_mask"][0].tolist()
            new_attn = old_attn + [1] * len(gen_ids)
            forward_kwargs["attention_mask"] = torch.tensor(
                [new_attn],
                dtype=forward_kwargs["attention_mask"].dtype,
                device=model.device,
            )

        outputs = model(**forward_kwargs)
        logits = outputs.logits[0]

        step_top_ids: List[np.ndarray] = []
        step_top_logprobs: List[np.ndarray] = []

        for s in range(len(gen_ids)):
            pos = input_len + s - 1
            step_logits = logits[pos].detach().float().cpu().numpy()
            top_ids, top_lps = topk_logprobs_from_logits(step_logits, baseline_topk)
            step_top_ids.append(top_ids)
            step_top_logprobs.append(top_lps)

        cache.append(BaselineSample(
            sample_id=str(rec["id"]),
            question=str(rec["question"]),
            answer=str(rec["answer"]),
            generated_ids=[int(x) for x in gen_ids],
            step_top_ids=step_top_ids,
            step_top_logprobs=step_top_logprobs,
        ))

        log_print(
            f"[rank {get_rank()} baseline {i+1:03d}/{len(records)}] sample_id={rec['id']} gen_len={len(gen_ids)}",
            main_only=False,
        )

    barrier()
    return cache


# ============================================================
# bandit state
# ============================================================

@dataclass
class PolicyResult:
    policy: List[int]
    mean_kl: float
    raw_reward: float
    skip_ratio: float
    skip_units: int
    per_layer_skip: List[float]


@dataclass
class BanditState:
    q_values: np.ndarray
    counts: np.ndarray
    reward_sums: np.ndarray
    history: List[PolicyResult]

    def refresh_q_values(self) -> None:
        mask = self.counts > 0
        self.q_values[mask] = self.reward_sums[mask] / self.counts[mask]

    def best_result(self) -> Optional[PolicyResult]:
        if not self.history:
            return None
        return max(self.history, key=lambda x: x.raw_reward)


# ============================================================
# sampling
# ============================================================

def softmax_np(x: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    z = x / t
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    return p


def row_probs_with_exploration(
    q_row: np.ndarray,
    count_row: np.ndarray,
    temperature: float,
    epsilon: float,
    low_count_boost: float,
) -> np.ndarray:
    base = softmax_np(q_row, temperature)

    explore = 1.0 / np.sqrt(count_row + 1.0)
    explore = explore / np.sum(explore)

    p = (1.0 - epsilon) * base + epsilon * explore

    if low_count_boost > 0:
        bonus = low_count_boost * (1.0 / np.sqrt(count_row + 1.0))
        p = p + bonus
        p = p / np.sum(p)

    return p


def sample_policy_exact_budget_monotone(
    q_values: np.ndarray,
    counts: np.ndarray,
    target_skip_units: int,
    rng: random.Random,
    temperature: float,
    epsilon: float,
    low_count_boost: float,
) -> List[int]:
    """
    Exact-budget sampling under monotone non-increasing constraint:
        k_1 >= k_2 >= ... >= k_L
    where each k_l is selected from {0,1,2,3,4,5,6}.
    """
    from functools import lru_cache

    L = q_values.shape[0]
    total_skip = int(target_skip_units)

    @lru_cache(None)
    def feasible(layer: int, remaining_skip: int, prev_k: int) -> bool:
        if layer == L:
            return remaining_skip == 0

        max_k = min(6, prev_k)
        for k in range(max_k, -1, -1):
            s = 6 - k
            if s > remaining_skip:
                continue
            if feasible(layer + 1, remaining_skip - s, k):
                return True
        return False

    if not feasible(0, total_skip, 6):
        raise RuntimeError(
            f"No feasible monotone policy for target_skip_units={target_skip_units}"
        )

    policy: List[int] = []
    remaining = total_skip
    prev_k = 6

    for l in range(L):
        q_row = q_values[l]
        count_row = counts[l]

        p_k = row_probs_with_exploration(
            q_row=q_row,
            count_row=count_row,
            temperature=temperature,
            epsilon=epsilon,
            low_count_boost=low_count_boost,
        )

        feasible_ks: List[int] = []
        feasible_ws: List[float] = []

        for k in range(prev_k, -1, -1):
            s = 6 - k
            if s > remaining:
                continue
            if feasible(l + 1, remaining - s, k):
                feasible_ks.append(k)
                feasible_ws.append(float(p_k[k]))

        if len(feasible_ks) == 0:
            raise RuntimeError(
                f"No feasible action at layer={l}, remaining={remaining}, prev_k={prev_k}"
            )

        ws = np.asarray(feasible_ws, dtype=np.float64)
        ws_sum = float(np.sum(ws))
        if ws_sum <= 0:
            ws = np.ones_like(ws, dtype=np.float64) / float(len(ws))
        else:
            ws = ws / ws_sum

        chosen_k = int(rng.choices(feasible_ks, weights=ws.tolist(), k=1)[0])
        policy.append(chosen_k)
        remaining -= (6 - chosen_k)
        prev_k = chosen_k

    if remaining != 0:
        raise RuntimeError(f"Monotone budget sampling failed: remaining={remaining}")

    return policy


def sample_policy_set_for_one_sample(
    q_values: np.ndarray,
    counts: np.ndarray,
    target_skip_units: int,
    rng: random.Random,
    temperature: float,
    epsilon: float,
    low_count_boost: float,
    group_size: int,
    seen: set,
    avoid_duplicates: bool,
    max_tries_per_policy: int,
) -> Tuple[List[List[int]], int]:
    sampled_policies: List[List[int]] = []
    local_seen = set()
    duplicate_fallbacks = 0

    for _ in range(group_size):
        sampled_policy = None
        for _try in range(max_tries_per_policy):
            candidate = sample_policy_exact_budget_monotone(
                q_values=q_values,
                counts=counts,
                target_skip_units=target_skip_units,
                rng=rng,
                temperature=temperature,
                epsilon=epsilon,
                low_count_boost=low_count_boost,
            )
            key = tuple(int(x) for x in candidate)
            if (not avoid_duplicates) or (key not in seen and key not in local_seen):
                sampled_policy = candidate
                local_seen.add(key)
                seen.add(key)
                break

        if sampled_policy is None:
            duplicate_fallbacks += 1
            sampled_policy = sample_policy_exact_budget_monotone(
                q_values=q_values,
                counts=counts,
                target_skip_units=target_skip_units,
                rng=rng,
                temperature=temperature,
                epsilon=epsilon,
                low_count_boost=low_count_boost,
            )
        sampled_policies.append(list(sampled_policy))

    return sampled_policies, duplicate_fallbacks



def enumerate_all_feasible_policies_monotone(
    n_layers: int,
    target_skip_units: int,
) -> List[List[int]]:
    from functools import lru_cache

    total_skip = int(target_skip_units)

    @lru_cache(None)
    def feasible(layer: int, remaining_skip: int, prev_k: int) -> bool:
        if layer == n_layers:
            return remaining_skip == 0
        max_k = min(6, prev_k)
        for k in range(max_k, -1, -1):
            s = 6 - k
            if s > remaining_skip:
                continue
            if feasible(layer + 1, remaining_skip - s, k):
                return True
        return False

    if not feasible(0, total_skip, 6):
        raise RuntimeError(f"No feasible monotone policy for target_skip_units={total_skip}")

    out: List[List[int]] = []
    cur: List[int] = []

    def dfs(layer: int, remaining_skip: int, prev_k: int) -> None:
        if layer == n_layers:
            if remaining_skip == 0:
                out.append(list(cur))
            return
        max_k = min(6, prev_k)
        for k in range(max_k, -1, -1):
            s = 6 - k
            if s > remaining_skip:
                continue
            if not feasible(layer + 1, remaining_skip - s, k):
                continue
            cur.append(int(k))
            dfs(layer + 1, remaining_skip - s, k)
            cur.pop()

    dfs(0, total_skip, 6)
    return out


def score_policy_from_matrix(
    q_values: np.ndarray,
    counts: np.ndarray,
    policy: Sequence[int],
    low_count_boost: float = 0.0,
) -> float:
    score = 0.0
    for l, k in enumerate(policy):
        k = int(k)
        score += float(q_values[l, k])
        if low_count_boost > 0:
            score += float(low_count_boost) / math.sqrt(float(counts[l, k]) + 1.0)
    return float(score)


def build_policy_pool_scores(
    all_policies: Sequence[Sequence[int]],
    available_policy_ids: Sequence[int],
    q_values: np.ndarray,
    counts: np.ndarray,
    temperature: float,
    epsilon: float,
    low_count_boost: float,
) -> np.ndarray:
    if len(available_policy_ids) == 0:
        return np.zeros((0,), dtype=np.float64)

    raw_scores = np.asarray([
        score_policy_from_matrix(
            q_values=q_values,
            counts=counts,
            policy=all_policies[pid],
            low_count_boost=low_count_boost,
        )
        for pid in available_policy_ids
    ], dtype=np.float64)

    probs_soft = softmax_np(raw_scores, temperature=max(float(temperature), 1e-6))
    probs_uni = np.ones_like(probs_soft, dtype=np.float64) / float(len(probs_soft))
    probs = (1.0 - float(epsilon)) * probs_soft + float(epsilon) * probs_uni
    probs = probs / np.sum(probs)
    return probs


def weighted_sample_without_replacement(
    candidate_ids: Sequence[int],
    probs: np.ndarray,
    k: int,
    rng: random.Random,
) -> List[int]:
    ids = list(int(x) for x in candidate_ids)
    weights = np.asarray(probs, dtype=np.float64).copy()
    out: List[int] = []
    k = min(int(k), len(ids))
    for _ in range(k):
        if len(ids) == 0:
            break
        wsum = float(np.sum(weights))
        if wsum <= 0:
            weights = np.ones_like(weights, dtype=np.float64) / float(len(weights))
        else:
            weights = weights / wsum
        chosen = int(rng.choices(ids, weights=weights.tolist(), k=1)[0])
        pos = ids.index(chosen)
        out.append(chosen)
        ids.pop(pos)
        if len(ids) == 0:
            break
        weights = np.delete(weights, pos)
    return out


def split_policy_ids_across_ranks(policy_ids: Sequence[int], world_size: int) -> List[List[int]]:
    out = [[] for _ in range(world_size)]
    for i, pid in enumerate(policy_ids):
        out[i % world_size].append(int(pid))
    return out


def sample_policy_ids_for_all_ranks(
    all_policies: Sequence[Sequence[int]],
    available_policy_ids: Sequence[int],
    q_values: np.ndarray,
    counts: np.ndarray,
    rng: random.Random,
    world_size: int,
    group_size: int,
    temperature: float,
    epsilon: float,
    low_count_boost: float,
) -> Tuple[List[List[int]], List[int]]:
    total_needed = int(world_size) * int(group_size)
    if len(available_policy_ids) == 0:
        return [[] for _ in range(world_size)], []

    probs = build_policy_pool_scores(
        all_policies=all_policies,
        available_policy_ids=available_policy_ids,
        q_values=q_values,
        counts=counts,
        temperature=temperature,
        epsilon=epsilon,
        low_count_boost=low_count_boost,
    )
    picked_from_available = weighted_sample_without_replacement(
        candidate_ids=available_policy_ids,
        probs=probs,
        k=min(total_needed, len(available_policy_ids)),
        rng=rng,
    )

    picked_ids = list(picked_from_available)
    remaining = int(total_needed) - len(picked_ids)

    # Refill shortage by sampling from the full policy set, ignoring whether it was
    # already consumed by the current available pool.
    if remaining > 0 and len(all_policies) > 0:
        full_policy_ids = list(range(len(all_policies)))
        refill_probs = build_policy_pool_scores(
            all_policies=all_policies,
            available_policy_ids=full_policy_ids,
            q_values=q_values,
            counts=counts,
            temperature=temperature,
            epsilon=epsilon,
            low_count_boost=low_count_boost,
        )
        refill_ids = weighted_sample_without_replacement(
            candidate_ids=full_policy_ids,
            probs=refill_probs,
            k=remaining,
            rng=rng,
        )
        picked_ids.extend(refill_ids)

    per_rank_policy_ids = split_policy_ids_across_ranks(picked_ids, world_size=world_size)
    return per_rank_policy_ids, list(picked_from_available)


def remove_policy_ids_from_available(
    available_policy_ids: Sequence[int],
    removed_policy_ids: Sequence[int],
) -> List[int]:
    removed = set(int(x) for x in removed_policy_ids)
    if not removed:
        return list(int(x) for x in available_policy_ids)
    return [int(x) for x in available_policy_ids if int(x) not in removed]


def gather_policy_eval_payloads(obj: Any) -> List[Any]:
    return all_gather_objects(obj)


# ============================================================
# reward update
# ============================================================

def compute_group_relative_advantages(
    mean_kls: Sequence[float],
    eps: float = 1e-6,
) -> List[float]:
    scores = -np.asarray(list(mean_kls), dtype=np.float64)
    if scores.size == 0:
        return []
    mu = float(np.mean(scores))
    std = float(np.std(scores))
    if std < eps:
        return [0.0 for _ in range(len(scores))]
    adv = (scores - mu) / (std + eps)
    return [float(x) for x in adv]


def accumulate_policy_reward_matrices(
    reward_sums: np.ndarray,
    reward_counts: np.ndarray,
    policy: Sequence[int],
    raw_reward: float,
) -> None:
    for l, k in enumerate(policy):
        k = int(k)
        reward_sums[l, k] += float(raw_reward)
        reward_counts[l, k] += 1.0


def solve_best_policy_from_matrix(
    score_matrix: np.ndarray,
    count_matrix: np.ndarray,
    target_skip_units: int,
    require_visited: bool = False,
) -> Tuple[List[int], float]:
    from functools import lru_cache

    L = score_matrix.shape[0]
    NEG = -1e100

    @lru_cache(None)
    def dp(layer: int, remaining_skip: int, prev_k: int) -> float:
        if layer == L:
            return 0.0 if remaining_skip == 0 else NEG

        best = NEG
        max_k = min(6, prev_k)
        for k in range(max_k, -1, -1):
            s = 6 - k
            if s > remaining_skip:
                continue
            if require_visited and count_matrix[layer, k] <= 0:
                continue
            tail = dp(layer + 1, remaining_skip - s, k)
            if tail <= NEG / 2:
                continue
            cand = float(score_matrix[layer, k]) + tail
            if cand > best:
                best = cand
        return best

    best_score = dp(0, int(target_skip_units), 6)
    if best_score <= NEG / 2:
        raise RuntimeError(
            "No feasible policy can be solved from final matrix under the current constraints"
        )

    policy: List[int] = []
    remaining = int(target_skip_units)
    prev_k = 6
    for layer in range(L):
        chosen_k = None
        chosen_score = NEG
        max_k = min(6, prev_k)
        for k in range(max_k, -1, -1):
            s = 6 - k
            if s > remaining:
                continue
            if require_visited and count_matrix[layer, k] <= 0:
                continue
            tail = dp(layer + 1, remaining - s, k)
            if tail <= NEG / 2:
                continue
            cand = float(score_matrix[layer, k]) + tail
            if cand > chosen_score:
                chosen_score = cand
                chosen_k = int(k)
        if chosen_k is None:
            raise RuntimeError(f"Backtracking failed at layer={layer}")
        policy.append(chosen_k)
        remaining -= (6 - chosen_k)
        prev_k = chosen_k

    return policy, float(best_score)


# ============================================================
# KL evaluation
# ============================================================

def eval_policy_on_records_local(
    model,
    processor,
    moe_layer_indices: List[int],
    policy: Sequence[int],
    records: List[Dict[str, Any]],
    baseline_cache: List[BaselineSample],
    sparse_topk: int,
) -> Tuple[List[float], float, List[float]]:
    import torch

    apply_policy_to_model(model, moe_layer_indices, policy)

    kimi.SKIP_EXP_COUNT = 0
    kimi.TOTAL_EXP_COUNT = 0
    kimi.SKIP_STATS = {}

    sample_kls: List[float] = []

    for rec, base in zip(records, baseline_cache):
        image = decode_image(rec["image"])
        batch = prepare_single_input(processor, image, rec["question"])
        batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}

        gen_ids = list(base.generated_ids)
        if len(gen_ids) == 0:
            sample_kls.append(0.0)
            continue

        input_len = int(batch["input_ids"].shape[1])
        full_ids = torch.tensor(
            [batch["input_ids"][0].tolist() + gen_ids],
            dtype=batch["input_ids"].dtype,
            device=model.device,
        )

        forward_kwargs = dict(batch)
        forward_kwargs["input_ids"] = full_ids
        if "attention_mask" in forward_kwargs:
            old_attn = forward_kwargs["attention_mask"][0].tolist()
            new_attn = old_attn + [1] * len(gen_ids)
            forward_kwargs["attention_mask"] = torch.tensor(
                [new_attn],
                dtype=forward_kwargs["attention_mask"].dtype,
                device=model.device,
            )

        with torch.no_grad():
            outputs = model(**forward_kwargs)

        logits = outputs.logits[0]
        step_kls = []
        for s in range(len(gen_ids)):
            pos = input_len + s - 1
            step_logits = logits[pos].detach().float().cpu().numpy()
            q_ids, q_lps = topk_logprobs_from_logits(step_logits, sparse_topk)
            p_ids = base.step_top_ids[s]
            p_lps = base.step_top_logprobs[s]
            step_kl = sparse_topk_kl(
                p_ids=p_ids,
                p_logprobs=p_lps,
                q_ids=q_ids,
                q_logprobs=q_lps,
                floor_logprob=-30.0,
            )
            step_kls.append(step_kl)
        sample_kls.append(float(np.mean(step_kls)) if len(step_kls) > 0 else 0.0)

    local_skip_exp = float(getattr(kimi, "SKIP_EXP_COUNT", 0.0))
    local_total_exp = float(getattr(kimi, "TOTAL_EXP_COUNT", 0.0))
    local_skip_ratio = local_skip_exp / max(local_total_exp, 1.0)

    skip_report = kimi.report_skip_stats(kimi.SKIP_STATS)
    local_per_layer_skip = [float(r["visual_prob"]) for r in skip_report.get("per_layer", [])]

    return sample_kls, float(local_skip_ratio), local_per_layer_skip


def reduce_final_policy_metrics(
    sample_kls: List[float],
    local_skip_ratio: float,
    local_per_layer_skip: List[float],
    device_str: str,
    world_size: int,
) -> Tuple[float, float, List[float]]:
    import torch

    device = torch.device(device_str)
    global_kl_sum = all_reduce_sum_scalar(float(np.sum(sample_kls)), device)
    global_count = all_reduce_sum_scalar(float(len(sample_kls)), device)
    mean_kl = global_kl_sum / max(global_count, 1.0)

    skip_ratio_sum = all_reduce_sum_scalar(float(local_skip_ratio), device)
    skip_ratio = skip_ratio_sum / float(world_size)

    if len(local_per_layer_skip) > 0:
        per_layer_sum = all_reduce_sum_vector(local_per_layer_skip, device)
        per_layer_skip = [x / float(world_size) for x in per_layer_sum]
    else:
        per_layer_skip = []

    return float(mean_kl), float(skip_ratio), per_layer_skip


def eval_policy_on_one_sample_from_local_cache(
    policy_local_kls: List[float],
    sample_index: int,
) -> float:
    return float(policy_local_kls[sample_index])


def evaluate_final_policy_global(
    model,
    processor,
    moe_layer_indices: List[int],
    policy: Sequence[int],
    records: List[Dict[str, Any]],
    baseline_cache: List[BaselineSample],
    sparse_topk: int,
    device_str: str,
    world_size: int,
) -> Dict[str, Any]:
    local_kls, local_skip_ratio, local_per_layer_skip = eval_policy_on_records_local(
        model=model,
        processor=processor,
        moe_layer_indices=moe_layer_indices,
        policy=policy,
        records=records,
        baseline_cache=baseline_cache,
        sparse_topk=sparse_topk,
    )
    mean_kl, skip_ratio, per_layer_skip = reduce_final_policy_metrics(
        sample_kls=local_kls,
        local_skip_ratio=local_skip_ratio,
        local_per_layer_skip=local_per_layer_skip,
        device_str=device_str,
        world_size=world_size,
    )
    barrier()
    return {
        "policy": list(policy),
        "mean_kl": float(mean_kl),
        "skip_ratio": float(skip_ratio),
        "skip_units": policy_skip_units(policy),
        "per_layer_skip": list(per_layer_skip),
    }



def summarize_local_policy_stats(policy_stats_local: Dict[Tuple[int, ...], Dict[str, Any]], topk: int = 32) -> Dict[str, Any]:
    items = []
    for st in policy_stats_local.values():
        avg_reward = float(st["reward_sum"] / max(st["reward_count"], 1))
        avg_kl = float(st["kl_sum"] / max(st["kl_count"], 1))
        avg_skip = float(st["skip_ratio_sum"] / max(st["skip_ratio_count"], 1))
        items.append({
            "policy": list(st["policy"]),
            "avg_reward": avg_reward,
            "avg_kl": avg_kl,
            "avg_skip_ratio": avg_skip,
            "skip_units": policy_skip_units(st["policy"]),
            "seen_count": int(st["reward_count"]),
            "per_layer_skip_last": list(st["per_layer_skip_last"]),
        })
    by_reward = sorted(items, key=lambda x: (-x["avg_reward"], x["avg_kl"]))[:topk]
    by_kl = sorted(items, key=lambda x: (x["avg_kl"], -x["avg_reward"]))[:topk]
    return {
        "num_unique": len(items),
        "top_by_reward": by_reward,
        "top_by_kl": by_kl,
    }


def merge_candidate_summaries(parts: List[Dict[str, Any]]) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    merged: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for part in parts:
        for key_name in ["top_by_reward", "top_by_kl"]:
            for item in part.get(key_name, []):
                key = tuple(int(x) for x in item["policy"])
                prev = merged.get(key)
                if prev is None:
                    merged[key] = dict(item)
                else:
                    # keep the better reward view and the better kl view conservatively
                    if item["avg_reward"] > prev["avg_reward"]:
                        prev.update(dict(item))
                    elif item["avg_kl"] < prev["avg_kl"]:
                        prev.update(dict(item))
                    prev["seen_count"] = max(int(prev.get("seen_count", 0)), int(item.get("seen_count", 0)))
    return merged


def sample_policy_exact_budget_unconstrained(
    q_values: np.ndarray,
    counts: np.ndarray,
    target_skip_units: int,
    rng: random.Random,
    temperature: float,
    epsilon: float,
    low_count_boost: float,
) -> List[int]:
    """
    Exact-budget sampling without monotonicity constraint.
    Each layer chooses k in {0,1,2,3,4,5,6}, with only the total skip budget enforced.
    """
    from functools import lru_cache

    L = q_values.shape[0]
    total_skip = int(target_skip_units)

    @lru_cache(None)
    def feasible(layer: int, remaining_skip: int) -> bool:
        if layer == L:
            return remaining_skip == 0
        remaining_layers = L - layer
        if remaining_skip < 0 or remaining_skip > 6 * remaining_layers:
            return False
        for k in range(6, -1, -1):
            s = 6 - k
            if s > remaining_skip:
                continue
            if feasible(layer + 1, remaining_skip - s):
                return True
        return False

    if not feasible(0, total_skip):
        raise RuntimeError(
            f"No feasible unconstrained policy for target_skip_units={target_skip_units}"
        )

    policy: List[int] = []
    remaining = total_skip

    for l in range(L):
        q_row = q_values[l]
        count_row = counts[l]

        p_k = row_probs_with_exploration(
            q_row=q_row,
            count_row=count_row,
            temperature=temperature,
            epsilon=epsilon,
            low_count_boost=low_count_boost,
        )

        feasible_ks: List[int] = []
        feasible_ws: List[float] = []

        for k in range(6, -1, -1):
            s = 6 - k
            if s > remaining:
                continue
            if feasible(l + 1, remaining - s):
                feasible_ks.append(k)
                feasible_ws.append(float(p_k[k]))

        if len(feasible_ks) == 0:
            raise RuntimeError(
                f"No feasible action at layer={l}, remaining={remaining}"
            )

        ws = np.asarray(feasible_ws, dtype=np.float64)
        ws_sum = float(np.sum(ws))
        if ws_sum <= 0:
            ws = np.ones_like(ws, dtype=np.float64) / float(len(ws))
        else:
            ws = ws / ws_sum

        chosen_k = int(rng.choices(feasible_ks, weights=ws.tolist(), k=1)[0])
        policy.append(chosen_k)
        remaining -= (6 - chosen_k)

    if remaining != 0:
        raise RuntimeError(f"Unconstrained budget sampling failed: remaining={remaining}")

    return policy


def split_policy_packets_across_ranks(
    sampled_policies: Sequence[Sequence[int]],
    world_size: int,
) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = [[] for _ in range(world_size)]
    for i, policy in enumerate(sampled_policies):
        out[i % world_size].append({
            "sample_slot": int(i),
            "policy": list(int(x) for x in policy),
        })
    return out


def sample_policy_packets_for_all_ranks(
    q_values: np.ndarray,
    counts: np.ndarray,
    target_skip_units: int,
    rng: random.Random,
    world_size: int,
    group_size: int,
    temperature: float,
    epsilon: float,
    low_count_boost: float,
    seen_policy_keys_global: set,
    avoid_duplicates: bool,
    max_tries_per_policy: int,
) -> Tuple[List[List[Dict[str, Any]]], int, int]:
    total_needed = int(world_size) * int(group_size)
    sampled_policies: List[List[int]] = []
    local_seen = set()
    duplicate_fallbacks = 0
    new_unique_count = 0

    for _ in range(total_needed):
        accepted = None
        last_candidate = None

        for _try in range(max_tries_per_policy):
            candidate = sample_policy_exact_budget_unconstrained(
                q_values=q_values,
                counts=counts,
                target_skip_units=target_skip_units,
                rng=rng,
                temperature=temperature,
                epsilon=epsilon,
                low_count_boost=low_count_boost,
            )
            key = tuple(int(x) for x in candidate)
            last_candidate = candidate

            if key in local_seen:
                continue
            if avoid_duplicates and key in seen_policy_keys_global:
                continue

            accepted = candidate
            local_seen.add(key)
            if avoid_duplicates:
                seen_policy_keys_global.add(key)
            new_unique_count += 1
            break

        if accepted is None:
            duplicate_fallbacks += 1
            for _try in range(max_tries_per_policy):
                candidate = sample_policy_exact_budget_unconstrained(
                    q_values=q_values,
                    counts=counts,
                    target_skip_units=target_skip_units,
                    rng=rng,
                    temperature=temperature,
                    epsilon=epsilon,
                    low_count_boost=low_count_boost,
                )
                key = tuple(int(x) for x in candidate)
                last_candidate = candidate
                if key in local_seen:
                    continue
                accepted = candidate
                local_seen.add(key)
                break

        if accepted is None:
            if last_candidate is None:
                last_candidate = sample_policy_exact_budget_unconstrained(
                    q_values=q_values,
                    counts=counts,
                    target_skip_units=target_skip_units,
                    rng=rng,
                    temperature=temperature,
                    epsilon=epsilon,
                    low_count_boost=low_count_boost,
                )
            accepted = last_candidate
            local_seen.add(tuple(int(x) for x in accepted))

        sampled_policies.append(list(int(x) for x in accepted))

    per_rank_policy_packets = split_policy_packets_across_ranks(sampled_policies, world_size=world_size)
    return per_rank_policy_packets, int(new_unique_count), int(duplicate_fallbacks)



# ============================================================
# main search
# ============================================================


def run_search(args):
    import torch

    run_start_dt = datetime.now()
    run_start_perf = time.perf_counter()

    rank, world_size, local_rank = setup_distributed()
    base_seed = int(args.seed)
    set_seed(base_seed + rank)
    rng = random.Random(base_seed + 100003 * rank)
    device_str = f"cuda:{local_rank}"

    all_records = build_records_from_two_parquets(
        inst_path=args.inst_parquet,
        img_path=args.images_parquet,
        limit=args.num_samples,
        inst_key_col=args.inst_key_col,
        img_key_col=args.img_key_col,
        question_col=args.question_col,
        answer_col=args.answer_col,
        image_col=args.image_col,
    )

    usable_global_samples = (len(all_records) // world_size) * world_size
    if usable_global_samples <= 0:
        raise RuntimeError(
            f"Not enough records for distributed run: loaded={len(all_records)} world_size={world_size}"
        )
    if usable_global_samples != len(all_records):
        if is_main_process():
            log_print(
                f"[data] trimming global samples from {len(all_records)} to {usable_global_samples} "
                "to keep per-rank batch counts identical"
            )
        all_records = all_records[:usable_global_samples]

    local_records = shard_records(all_records, world_size=world_size, rank=rank)

    temp_model, temp_processor = kimi.load_model(
        model_path=args.model_path,
        layer_gate_dict=None,
        device_map="cpu",
    )

    moe_layer_indices = discover_moe_layer_indices(temp_model)
    n_moe_layers = len(moe_layer_indices)
    if n_moe_layers <= 0:
        raise RuntimeError("No MoE layers discovered")

    avg_visual_tokens, avg_text_tokens = infer_token_profile(
        all_records,
        temp_processor,
        media_placeholder_token_id=int(temp_model.config.media_placeholder_token_id),
        special_token_ids=list(temp_processor.tokenizer.all_special_ids),
    )

    fixed_text_tokens = float(args.fixed_text_token_count) if float(args.fixed_text_token_count) > 0 else float(avg_text_tokens)
    fixed_visual_tokens = float(args.fixed_visual_token_count) if float(args.fixed_visual_token_count) > 0 else float(avg_visual_tokens)

    target_skip_units = compute_target_skip_units(
        target_skip_ratio=args.target_skip_ratio,
        n_moe_layers=n_moe_layers,
        avg_visual_tokens=fixed_visual_tokens,
        avg_text_tokens=fixed_text_tokens,
    )

    pool_metadata = {
        "strategy_pool_removed": True,
        "search_monotone": False,
        "final_select_monotone": True,
    }

    try:
        del temp_model
        del temp_processor
        torch.cuda.empty_cache()
    except Exception:
        pass

    log_print(
        f"[dist] rank={rank} local_rank={local_rank} world_size={world_size} device={device_str}",
        main_only=False,
    )
    log_print(
        f"[data] fixed global samples={len(all_records)} | local shard on rank {rank} = {len(local_records)}",
        main_only=False,
    )
    if is_main_process():
        log_print(
            f"[token_profile] inferred_text={avg_text_tokens:.4f} inferred_visual={avg_visual_tokens:.4f} "
            f"used_fixed_text={fixed_text_tokens:.4f} used_fixed_visual={fixed_visual_tokens:.4f}"
        )

    log_print("===== Loading FULL model onto GPU =====")
    model, processor = kimi.load_model(
        model_path=args.model_path,
        layer_gate_dict=None,
        device_map=device_str,
    )
    model.eval()
    apply_policy_to_model(model, moe_layer_indices, [6] * n_moe_layers)

    log_print("===== Build baseline cache =====")
    baseline_cache = build_baseline_cache(
        model=model,
        processor=processor,
        records=local_records,
        max_new_tokens=args.max_new_tokens,
        baseline_topk=args.baseline_topk,
    )

    state = BanditState(
        q_values=np.zeros((n_moe_layers, 7), dtype=np.float64),
        counts=np.zeros((n_moe_layers, 7), dtype=np.float64),
        reward_sums=np.zeros((n_moe_layers, 7), dtype=np.float64),
        history=[],
    )

    policy_stats_local: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    rounds_completed = 0
    stop_reason = "max_rounds_reached"
    batch_size = args.batch_size
    num_batches = math.ceil(len(local_records) / batch_size) if len(local_records) > 0 else 0
    num_total_unique_sampled = 0
    total_policy_samples_selected_global = 0
    seen_policy_keys_global: set = set()
    duplicate_fallbacks_global = 0

    for round_idx in range(args.max_rounds):
        round_new_unique_local = 0
        round_sample_groups_local = 0

        for batch_idx in range(num_batches):
            step_reward_sums = np.zeros_like(state.reward_sums)
            step_reward_counts = np.zeros_like(state.counts)
            step_new_unique_local = 0
            step_sample_groups_local = 0

            batch_start = batch_idx * batch_size
            batch_records = local_records[batch_start : batch_start + batch_size]
            batch_baselines = baseline_cache[batch_start : batch_start + batch_size]

            if is_main_process():
                per_rank_policy_packets, new_unique_global_step, duplicate_fallbacks_step = sample_policy_packets_for_all_ranks(
                    q_values=state.q_values,
                    counts=state.counts,
                    target_skip_units=target_skip_units,
                    rng=rng,
                    world_size=world_size,
                    group_size=args.group_size,
                    temperature=args.temperature,
                    epsilon=args.epsilon,
                    low_count_boost=args.low_count_boost,
                    seen_policy_keys_global=seen_policy_keys_global,
                    avoid_duplicates=bool(args.avoid_duplicates),
                    max_tries_per_policy=int(args.max_tries_per_policy),
                )
                total_policy_samples_selected_global += int(world_size) * int(args.group_size)
                duplicate_fallbacks_global += int(duplicate_fallbacks_step)
            else:
                per_rank_policy_packets = None
                new_unique_global_step = None
                duplicate_fallbacks_step = None

            per_rank_policy_packets = broadcast_object(per_rank_policy_packets, src=0)
            new_unique_global_step = int(broadcast_object(new_unique_global_step, src=0))
            duplicate_fallbacks_step = int(broadcast_object(duplicate_fallbacks_step, src=0))

            my_policy_packets = list(per_rank_policy_packets[rank])
            sampled_policies = [list(pkt["policy"]) for pkt in my_policy_packets]

            step_new_unique_local += len(my_policy_packets)
            step_sample_groups_local += 1 if len(my_policy_packets) > 0 else 0

            local_payloads = []
            for pkt in my_policy_packets:
                policy = list(pkt["policy"])
                sample_kls, local_skip_ratio, local_per_layer_skip = eval_policy_on_records_local(
                    model=model,
                    processor=processor,
                    moe_layer_indices=moe_layer_indices,
                    policy=policy,
                    records=batch_records,
                    baseline_cache=batch_baselines,
                    sparse_topk=args.eval_topk,
                )
                batch_mean_kl = float(np.mean(sample_kls)) if len(sample_kls) > 0 else 0.0
                local_payloads.append({
                    "sample_slot": int(pkt["sample_slot"]),
                    "policy": list(policy),
                    "mean_kl": float(batch_mean_kl),
                    "skip_ratio": float(local_skip_ratio),
                    "per_layer_skip": list(local_per_layer_skip),
                })

            gathered_payloads = gather_policy_eval_payloads(local_payloads)

            if is_main_process():
                flat_payloads = []
                for part in gathered_payloads:
                    flat_payloads.extend(part)

                ordered_payloads = sorted(flat_payloads, key=lambda x: x["sample_slot"])
                ordered_mean_kls = [float(x["mean_kl"]) for x in ordered_payloads]
                group_advantages = compute_group_relative_advantages(ordered_mean_kls)

                per_rank_rewards: List[List[Dict[str, Any]]] = [[] for _ in range(world_size)]
                slot_to_rank: Dict[int, int] = {}
                for r, packets in enumerate(per_rank_policy_packets):
                    for pkt in packets:
                        slot_to_rank[int(pkt["sample_slot"])] = int(r)

                for i, payload in enumerate(ordered_payloads):
                    raw_reward = float(group_advantages[i])
                    policy = list(payload["policy"])
                    accumulate_policy_reward_matrices(
                        reward_sums=step_reward_sums,
                        reward_counts=step_reward_counts,
                        policy=policy,
                        raw_reward=raw_reward,
                    )
                    dst_rank = slot_to_rank[int(payload["sample_slot"])]
                    per_rank_rewards[dst_rank].append({
                        "sample_slot": int(payload["sample_slot"]),
                        "policy": list(policy),
                        "raw_reward": raw_reward,
                        "mean_kl": float(payload["mean_kl"]),
                        "skip_ratio": float(payload["skip_ratio"]),
                        "per_layer_skip": list(payload["per_layer_skip"]),
                    })
                    state.history.append(PolicyResult(
                        policy=list(policy),
                        mean_kl=float(payload["mean_kl"]),
                        raw_reward=raw_reward,
                        skip_ratio=float(payload["skip_ratio"]),
                        skip_units=policy_skip_units(policy),
                        per_layer_skip=list(payload["per_layer_skip"]),
                    ))
            else:
                per_rank_rewards = None

            per_rank_rewards = broadcast_object(per_rank_rewards, src=0)
            my_rewards = per_rank_rewards[rank]
            for item in my_rewards:
                key = tuple(int(x) for x in item["policy"])
                if key not in policy_stats_local:
                    policy_stats_local[key] = {
                        "policy": list(item["policy"]),
                        "reward_sum": 0.0,
                        "reward_count": 0,
                        "kl_sum": 0.0,
                        "kl_count": 0,
                        "skip_ratio_sum": 0.0,
                        "skip_ratio_count": 0,
                        "per_layer_skip_last": list(item["per_layer_skip"]),
                    }
                st = policy_stats_local[key]
                st["reward_sum"] += float(item["raw_reward"])
                st["reward_count"] += 1
                st["kl_sum"] += float(item["mean_kl"])
                st["kl_count"] += 1
                st["skip_ratio_sum"] += float(item["skip_ratio"])
                st["skip_ratio_count"] += 1
                st["per_layer_skip_last"] = list(item["per_layer_skip"])

            device = torch.device(device_str)
            global_step_reward_sums = np.asarray(
                all_reduce_sum_vector(step_reward_sums.reshape(-1).tolist(), device), dtype=np.float64
            ).reshape(n_moe_layers, 7)
            global_step_reward_counts = np.asarray(
                all_reduce_sum_vector(step_reward_counts.reshape(-1).tolist(), device), dtype=np.float64
            ).reshape(n_moe_layers, 7)

            state.reward_sums += global_step_reward_sums
            state.counts += global_step_reward_counts
            state.refresh_q_values()

            global_step_new_unique = int(all_reduce_sum_scalar(step_new_unique_local, device))
            global_step_sample_groups = int(all_reduce_sum_scalar(step_sample_groups_local, device))

            round_new_unique_local += step_new_unique_local
            round_sample_groups_local += step_sample_groups_local
            num_total_unique_sampled += int(new_unique_global_step)

            if is_main_process() and args.log_sample_every > 0 and ((batch_idx + 1) % args.log_sample_every == 0):
                log_print(
                    f"[round {round_idx+1:02d} step {batch_idx+1:03d}] groups={global_step_sample_groups} "
                    f"new_unique_local_eval={global_step_new_unique} sampled_new_unique={new_unique_global_step} "
                    f"duplicate_fallbacks_step={duplicate_fallbacks_step} total_samples={total_policy_samples_selected_global}"
                )

            barrier()

            if is_main_process():
                should_stop_no_new = (
                    args.avoid_duplicates
                    and int(new_unique_global_step) <= 0
                    and not (
                        int(args.min_total_policy_samples) > 0
                        and int(total_policy_samples_selected_global) < int(args.min_total_policy_samples)
                    )
                )
            else:
                should_stop_no_new = None

            should_stop_no_new = bool(broadcast_object(should_stop_no_new, src=0))
            if should_stop_no_new:
                stop_reason = "no_new_policy_available"
                break

        rounds_completed = round_idx + 1

        device = torch.device(device_str)
        global_round_new_unique = int(all_reduce_sum_scalar(round_new_unique_local, device))
        global_round_sample_groups = int(all_reduce_sum_scalar(round_sample_groups_local, device))

        if is_main_process():
            log_print(
                f"[round {round_idx+1:02d}] groups={global_round_sample_groups} "
                f"new_unique_local_eval={global_round_new_unique} total_samples={total_policy_samples_selected_global} "
                f"global_seen={len(seen_policy_keys_global)} duplicate_fallbacks_global={duplicate_fallbacks_global}"
            )
            if args.print_q_every > 0 and ((round_idx + 1) % args.print_q_every == 0):
                log_print("Current Q-values:")
                log_print(np.array2string(state.q_values, precision=4, suppress_small=True))
                log_print("Current counts:")
                log_print(np.array2string(state.counts, precision=0, suppress_small=True))

        barrier()

        if stop_reason == "no_new_policy_available":
            break

        if is_main_process():
            should_stop_min_samples = (
                int(args.min_total_policy_samples) > 0
                and int(total_policy_samples_selected_global) >= int(args.min_total_policy_samples)
            )
        else:
            should_stop_min_samples = None

        should_stop_min_samples = bool(broadcast_object(should_stop_min_samples, src=0))
        if should_stop_min_samples:
            stop_reason = "min_total_policy_samples_reached"
            break

    local_summary = summarize_local_policy_stats(policy_stats_local, topk=getattr(args, "candidate_topk", 32))
    gathered_summaries = all_gather_objects(local_summary)

    if is_main_process():
        merged_candidates = merge_candidate_summaries(gathered_summaries)
        num_unique_policies_seen = int(sum(int(x.get("num_unique", 0)) for x in gathered_summaries))

        best_seen_by_reward = None
        best_seen_by_kl = None
        for item in merged_candidates.values():
            if best_seen_by_reward is None or item["avg_reward"] > best_seen_by_reward["avg_reward"]:
                best_seen_by_reward = dict(item)
            if best_seen_by_kl is None or item["avg_kl"] < best_seen_by_kl["avg_kl"]:
                best_seen_by_kl = dict(item)

        matrix_policy, matrix_objective = solve_best_policy_from_matrix(
            score_matrix=state.q_values,
            count_matrix=state.counts,
            target_skip_units=target_skip_units,
            require_visited=False,
        )
        current_style_policy = list(best_seen_by_reward["policy"]) if best_seen_by_reward is not None else list(matrix_policy)
    else:
        matrix_policy = None
        current_style_policy = None
        matrix_objective = None
        best_seen_by_reward = None
        best_seen_by_kl = None
        num_unique_policies_seen = None

    matrix_policy = broadcast_object(matrix_policy, src=0)
    current_style_policy = broadcast_object(current_style_policy, src=0)

    matrix_eval = evaluate_final_policy_global(
        model=model,
        processor=processor,
        moe_layer_indices=moe_layer_indices,
        policy=matrix_policy,
        records=local_records,
        baseline_cache=baseline_cache,
        sparse_topk=args.eval_topk,
        device_str=device_str,
        world_size=world_size,
    )
    current_style_eval = evaluate_final_policy_global(
        model=model,
        processor=processor,
        moe_layer_indices=moe_layer_indices,
        policy=current_style_policy,
        records=local_records,
        baseline_cache=baseline_cache,
        sparse_topk=args.eval_topk,
        device_str=device_str,
        world_size=world_size,
    )

    run_end_dt = datetime.now()
    elapsed_seconds = time.perf_counter() - run_start_perf
    elapsed_human = format_elapsed_human(elapsed_seconds)

    device = torch.device(device_str)
    global_num_total_unique_sampled = int(all_reduce_sum_scalar(float(num_total_unique_sampled), device))

    if is_main_process():
        print("\n===== Search finished =====", flush=True)
        print(
            f"Run time | start: {run_start_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"end: {run_end_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"elapsed: {elapsed_human}",
            flush=True,
        )
        print(json.dumps({
            "distributed": {
                "world_size": world_size,
                "rank": rank,
                "local_rank": local_rank,
            },
            "run_timing": {
                "start_time": run_start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": run_end_dt.strftime('%Y-%m-%d %H:%M:%S'),
                "elapsed_human": elapsed_human,
                "elapsed_seconds": float(elapsed_seconds),
            },
            "num_samples_global": len(all_records),
            "samples_per_rank": {str(r): len(shard_records(all_records, world_size, r)) for r in range(world_size)},
            "num_moe_layers": n_moe_layers,
            "moe_layer_indices": moe_layer_indices,
            "avg_visual_tokens_inferred": avg_visual_tokens,
            "avg_text_tokens_inferred": avg_text_tokens,
            "fixed_visual_token_count_used": float(fixed_visual_tokens),
            "fixed_text_token_count_used": float(fixed_text_tokens),
            "target_skip_ratio": args.target_skip_ratio,
            "target_skip_units": target_skip_units,
            "group_size_per_sample": args.group_size,
            "max_rounds": args.max_rounds,
            "rounds_completed": rounds_completed,
            "stop_reason": stop_reason,
            "online_sync_per_sample": True,
            "sampling_mode": "rank0_online_budget_only_unconstrained_sampling",
            "num_policy_evals": int(sum(int(x.get("reward_count", 0)) for x in merged_candidates.values())) if best_seen_by_reward is not None else 0,
            "num_unique_policies_seen_local_sum": num_unique_policies_seen,
            "num_total_unique_sampled_all_ranks": int(global_num_total_unique_sampled),
            "total_policy_samples_selected_global": int(total_policy_samples_selected_global),
            "min_total_policy_samples": int(args.min_total_policy_samples),
            "duplicate_fallbacks_global": int(duplicate_fallbacks_global),
            "policy_pool": {
                **pool_metadata,
                "global_seen_unique": int(len(seen_policy_keys_global)),
                "allow_revisit_after_pool_exhausted": bool(args.allow_revisit_after_pool_exhausted),
            },
            "matrix_solved_policy": {
                **matrix_eval,
                "matrix_objective": float(matrix_objective),
            },
            "current_code_style_policy": current_style_eval,
            "best_seen_policy_by_avg_reward": best_seen_by_reward,
            "best_seen_policy_by_avg_kl": best_seen_by_kl,
            "final_q_values": state.q_values.tolist(),
            "final_counts": state.counts.tolist(),
            "final_reward_sums": state.reward_sums.tolist(),
        }, ensure_ascii=False, indent=2), flush=True)

    barrier()


# ============================================================
# args
# ============================================================
# args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model-path", type=str, required=True)

    p.add_argument("--images-parquet", type=str,
                   default="dataset/GQA/testdev_balanced_images/testdev_balanced_images/testdev-00000-of-00001.parquet")
    p.add_argument("--inst-parquet", type=str,
                   default="dataset/GQA/testdev_balanced_instructions/testdev_balanced_instructions/testdev-00000-of-00001.parquet")

    p.add_argument("--inst-key-col", type=str, default="imageId")
    p.add_argument("--img-key-col", type=str, default="id")
    p.add_argument("--question-col", type=str, default="question")
    p.add_argument("--answer-col", type=str, default="answer")
    p.add_argument("--image-col", type=str, default="image")

    p.add_argument("--freq-path", type=str, default=None)

    p.add_argument("--num-samples", type=int, default=3000,
                   help="Global fixed sample count. In multi-GPU, all ranks together evaluate exactly these 100 samples.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Number of samples per batch for policy evaluation.")
    p.add_argument("--group-size", type=int, default=16,
                   help="Number of sampled policies for each single sample in one round.")
    p.add_argument("--max-rounds", type=int, default=1,
                   help="One round = all fixed samples each sample sampling group_size policies. Stop at max_rounds or when no new policy is available.")
    p.add_argument("--target-skip-ratio", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=8)

    p.add_argument("--baseline-topk", type=int, default=64)
    p.add_argument("--eval-topk", type=int, default=64)


    p.add_argument("--temperature", type=float, default=0.8,
                   help="Softmax temperature over per-layer action Q-values.")
    p.add_argument("--epsilon", type=float, default=0.10,
                   help="Mix-in exploration weight; exploration prefers low-count actions.")
    p.add_argument("--low-count-boost", type=float, default=0.02,
                   help="Extra probability mass toward rarely visited actions.")

    p.add_argument("--avoid-duplicates", action="store_true", default=True)
    p.add_argument("--allow-revisit-after-pool-exhausted", action="store_true")
    p.add_argument("--min-total-policy-samples", type=int, default=0,
                   help="Minimum total sampled policy count across all ranks. If pool is exhausted before reaching this count, policies are repeatedly reused from the full pool.")
    p.add_argument("--policy-pool-max-size", type=int, default=0,
                   help="0 means enumerate all feasible policies. Positive value means keep only the first N enumerated policies.")
    p.add_argument("--print-q-every", type=int, default=1)
    p.add_argument("--log-sample-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed-text-token-count", type=float, default=337438.0,
                   help="Use this fixed text-token count when converting target skip ratio to skip units. Set <=0 to fall back to inferred counts.")
    p.add_argument("--fixed-visual-token-count", type=float, default=4667061.0,
                   help="Use this fixed visual-token count when converting target skip ratio to skip units. Set <=0 to fall back to inferred counts.")
    p.add_argument("--max-tries-per-policy", type=int, default=64,
                   help="Maximum resampling attempts per policy before duplicate fallback is allowed.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search(args)
