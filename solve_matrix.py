import argparse
import math
import re
from typing import Any, Dict, List, Tuple


def compute_target_sum_k(
    target_overall_skip: float,
    count_text: int,
    count_visual: int,
    num_layers: int,
    max_experts: int,
) -> float:
    total_tokens = count_text + count_visual
    if total_tokens <= 0:
        raise ValueError("count_text + count_visual must be positive.")

    visual_weight = count_visual / total_tokens
    if visual_weight <= 0:
        raise ValueError("count_visual must be positive.")

    required_visual_skip = target_overall_skip / visual_weight
    required_visual_skip = max(0.0, min(1.0, required_visual_skip))

    total_budget = num_layers * max_experts
    return total_budget * (1.0 - required_visual_skip)


def evaluate_policy(
    policy: List[int],
    count_text: int,
    count_visual: int,
    max_experts: int,
) -> Dict[str, float]:
    num_layers = len(policy)
    raw_sum = sum(policy)
    total_visual_capacity = num_layers * max_experts
    visual_skip = 1.0 - raw_sum / total_visual_capacity

    total = count_text + count_visual
    overall_skip = visual_skip * (count_visual / total) if total > 0 else 0.0

    return {
        "raw_sum": raw_sum,
        "visual_skip": visual_skip,
        "overall_skip": overall_skip,
    }


def compress_pattern(policy: List[int], start_layer_index: int = 1) -> str:
    if not policy:
        return ""

    parts = []
    n = len(policy)
    seg_start = 0

    for i in range(1, n + 1):
        if i == n or policy[i] != policy[seg_start]:
            left = seg_start + start_layer_index
            right = i - 1 + start_layer_index
            value = policy[seg_start]
            if left == right:
                parts.append(f"{left}:{value}")
            else:
                parts.append(f"{left}-{right}:{value}")
            seg_start = i

    return "__".join(parts)


def solve_policy_no_monotonic(
    q_matrix: List[List[float]],
    target_sum_k_int: int,
) -> Tuple[List[int], float]:
    num_layers = len(q_matrix)
    num_actions = len(q_matrix[0])
    neg_inf = -1e100

    dp = [[neg_inf] * (target_sum_k_int + 1) for _ in range(num_layers + 1)]
    parent = [[-1] * (target_sum_k_int + 1) for _ in range(num_layers + 1)]
    dp[0][0] = 0.0

    for i in range(1, num_layers + 1):
        for total_k in range(target_sum_k_int + 1):
            best_val = neg_inf
            best_k = -1
            for k in range(num_actions):
                if total_k >= k and dp[i - 1][total_k - k] > neg_inf / 2:
                    cand = dp[i - 1][total_k - k] + q_matrix[i - 1][k]
                    if cand > best_val:
                        best_val = cand
                        best_k = k
            dp[i][total_k] = best_val
            parent[i][total_k] = best_k

    if dp[num_layers][target_sum_k_int] <= neg_inf / 2:
        raise RuntimeError("No feasible solution found without monotonicity.")

    policy = [0] * num_layers
    total_k = target_sum_k_int
    for i in range(num_layers, 0, -1):
        k = parent[i][total_k]
        if k < 0:
            raise RuntimeError("Backtracking failed without monotonicity.")
        policy[i - 1] = k
        total_k -= k

    return policy, dp[num_layers][target_sum_k_int]


def solve_policy_monotonic(
    q_matrix: List[List[float]],
    target_sum_k_int: int,
) -> Tuple[List[int], float]:
    num_layers = len(q_matrix)
    num_actions = len(q_matrix[0])
    neg_inf = -1e100

    dp_prev = [[neg_inf] * num_actions for _ in range(target_sum_k_int + 1)]
    parent = [
        [[-1] * num_actions for _ in range(target_sum_k_int + 1)]
        for _ in range(num_layers)
    ]

    for k in range(num_actions):
        if k <= target_sum_k_int:
            dp_prev[k][k] = q_matrix[0][k]
            parent[0][k][k] = -2

    for i in range(1, num_layers):
        dp_curr = [[neg_inf] * num_actions for _ in range(target_sum_k_int + 1)]

        for total_k in range(target_sum_k_int + 1):
            for prev_k in range(num_actions):
                prev_val = dp_prev[total_k][prev_k]
                if prev_val <= neg_inf / 2:
                    continue

                for k in range(prev_k + 1):
                    next_total_k = total_k + k
                    if next_total_k > target_sum_k_int:
                        continue
                    cand = prev_val + q_matrix[i][k]
                    if cand > dp_curr[next_total_k][k]:
                        dp_curr[next_total_k][k] = cand
                        parent[i][next_total_k][k] = prev_k

        dp_prev = dp_curr

    best_score = neg_inf
    best_last_k = -1
    for last_k in range(num_actions):
        if dp_prev[target_sum_k_int][last_k] > best_score:
            best_score = dp_prev[target_sum_k_int][last_k]
            best_last_k = last_k

    if best_last_k < 0:
        raise RuntimeError("No feasible solution found with monotonicity.")

    policy = [0] * num_layers
    total_k = target_sum_k_int
    k = best_last_k
    for i in range(num_layers - 1, -1, -1):
        policy[i] = k
        prev_k = parent[i][total_k][k]
        total_k -= k
        if i > 0:
            if prev_k < 0:
                raise RuntimeError("Backtracking failed with monotonicity.")
            k = prev_k

    return policy, best_score


def solve_from_q_matrix(
    q_matrix: List[List[float]],
    count_text: int,
    count_visual: int,
    target_overall_skip: float,
    monotonic: bool = False,
    start_layer_index: int = 1,
    prefer: str = "closest_then_best",
) -> Dict[str, Any]:
    if not q_matrix or not q_matrix[0]:
        raise ValueError("q_matrix is empty.")

    num_layers = len(q_matrix)
    num_actions = len(q_matrix[0])

    for row in q_matrix:
        if len(row) != num_actions:
            raise ValueError("All rows in q_matrix must have the same length.")

    max_experts = num_actions - 1
    target_sum_k_float = compute_target_sum_k(
        target_overall_skip=target_overall_skip,
        count_text=count_text,
        count_visual=count_visual,
        num_layers=num_layers,
        max_experts=max_experts,
    )

    floor_budget = max(0, min(num_layers * max_experts, math.floor(target_sum_k_float)))
    ceil_budget = max(0, min(num_layers * max_experts, math.ceil(target_sum_k_float)))
    candidate_budgets = sorted(set([floor_budget, ceil_budget]))

    candidates = []
    for budget in candidate_budgets:
        if monotonic:
            policy, score = solve_policy_monotonic(q_matrix, budget)
        else:
            policy, score = solve_policy_no_monotonic(q_matrix, budget)

        stats = evaluate_policy(policy, count_text, count_visual, max_experts)
        distance_to_target = abs(stats["overall_skip"] - target_overall_skip)

        candidates.append({
            "policy": policy,
            "score": score,
            "budget": budget,
            "distance_to_target": distance_to_target,
            **stats,
        })

    if prefer == "best_score":
        best = max(candidates, key=lambda x: x["score"])
    else:
        best = sorted(candidates, key=lambda x: (x["distance_to_target"], -x["score"]))[0]

    return {
        "policy": best["policy"],
        "pattern": compress_pattern(best["policy"], start_layer_index=start_layer_index),
        "score": best["score"],
        "raw_sum": best["raw_sum"],
        "visual_skip": best["visual_skip"],
        "overall_skip": best["overall_skip"],
        "num_layers": num_layers,
        "max_experts": max_experts,
        "target_sum_k_float": target_sum_k_float,
        "chosen_budget": best["budget"],
        "monotonic": monotonic,
    }


def extract_matrix_from_log(log_file_path: str) -> List[List[float]]:
    with open(log_file_path, "r", encoding="utf-8") as log_file:
        log_content = log_file.read()

    matrix_match = re.search(r"\[\[.*?\]\]", log_content, re.DOTALL)
    if not matrix_match:
        raise ValueError("No matrix found in the log file.")

    matrix_str = matrix_match.group(0).strip()
    matrix = []
    for row in matrix_str.split("\n"):
        row = row.strip()
        if row:
            row = row.replace("[", "").replace("]", "").strip()
            matrix.append(list(map(float, row.split())))

    return matrix


def solve_from_log_file(
    log_file_path: str,
    count_text: int,
    count_visual: int,
    target_overall_skip: float,
    monotonic: bool = False,
    start_layer_index: int = 1,
    prefer: str = "closest_then_best",
) -> Dict[str, Any]:
    q_matrix = extract_matrix_from_log(log_file_path)
    return solve_from_q_matrix(
        q_matrix=q_matrix,
        count_text=count_text,
        count_visual=count_visual,
        target_overall_skip=target_overall_skip,
        monotonic=monotonic,
        start_layer_index=start_layer_index,
        prefer=prefer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--skip-ratio", type=float, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count_text = 337438
    count_visual = 4667061

    result_no_monotonic = solve_from_log_file(
        log_file_path=args.log_file,
        count_text=count_text,
        count_visual=count_visual,
        target_overall_skip=args.skip_ratio,
        monotonic=False,
    )
    print("nomo result:")
    print(result_no_monotonic)

    result_monotonic = solve_from_log_file(
        log_file_path=args.log_file,
        count_text=count_text,
        count_visual=count_visual,
        target_overall_skip=args.skip_ratio,
        monotonic=True,
    )
    print("mono result:")
    print(result_monotonic)
