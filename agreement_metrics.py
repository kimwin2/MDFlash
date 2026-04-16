import math
from collections import Counter
from typing import Any

import torch


def build_batch_agreement_snapshot(draft_logits: torch.Tensor) -> dict[str, Any] | None:
    if draft_logits.dim() != 3 or draft_logits.shape[0] <= 1 or draft_logits.shape[1] == 0:
        return None

    top1_tokens = torch.argmax(draft_logits, dim=-1).to(device="cpu", dtype=torch.long)
    branch_count = int(top1_tokens.shape[0])
    depth_count = int(top1_tokens.shape[1])
    top1_by_branch = top1_tokens.tolist()

    majority_agreement = []
    base_agreement = []
    unique_counts = []
    majority_tokens = []
    base_tokens = top1_by_branch[0]

    for depth_idx in range(depth_count):
        depth_tokens = [int(top1_by_branch[branch_idx][depth_idx]) for branch_idx in range(branch_count)]
        token_counts = Counter(depth_tokens)
        majority_token, majority_count = token_counts.most_common(1)[0]
        base_token = int(base_tokens[depth_idx])

        majority_agreement.append(float(majority_count / branch_count))
        base_agreement.append(float(token_counts[base_token] / branch_count))
        unique_counts.append(int(len(token_counts)))
        majority_tokens.append(int(majority_token))

    return {
        "branch_count": branch_count,
        "depth_count": depth_count,
        "majority_agreement": majority_agreement,
        "base_agreement": base_agreement,
        "unique_counts": unique_counts,
        "majority_tokens": majority_tokens,
        "base_tokens": [int(token_id) for token_id in base_tokens],
    }


def append_batch_agreement_metric(
    batch_agreement_metrics: list[dict[str, Any]] | None,
    draft_logits: torch.Tensor,
    accepted_indices: list[int],
) -> None:
    if batch_agreement_metrics is None:
        return

    snapshot = build_batch_agreement_snapshot(draft_logits)
    if snapshot is None:
        return

    accepted_draft_tokens = max(0, min(len(accepted_indices) - 1, snapshot["depth_count"]))
    snapshot["acceptance_length"] = int(len(accepted_indices))
    snapshot["accepted_draft_tokens"] = int(accepted_draft_tokens)
    batch_agreement_metrics.append(snapshot)


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    denom_x = sum(x * x for x in centered_x)
    denom_y = sum(y * y for y in centered_y)
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / math.sqrt(denom_x * denom_y)


def summarize_batch_agreement_metrics(metrics: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not metrics:
        return {
            "rounds": 0,
            "tokens": 0,
            "avg_majority_agreement": None,
            "avg_base_agreement": None,
            "avg_accepted_draft_tokens": None,
            "token_majority_pearson": None,
            "token_base_pearson": None,
            "round_first_majority_pearson": None,
            "round_mean_majority_pearson": None,
        }

    token_majority_x = []
    token_base_x = []
    token_y = []
    round_first_majority_x = []
    round_mean_majority_x = []
    round_y = []
    all_majority = []
    all_base = []

    for metric in metrics:
        majority_agreement = [float(value) for value in metric.get("majority_agreement", [])]
        base_agreement = [float(value) for value in metric.get("base_agreement", [])]
        if not majority_agreement:
            continue

        accepted_draft_tokens = int(metric.get("accepted_draft_tokens", 0))
        depth_count = len(majority_agreement)
        all_majority.extend(majority_agreement)
        all_base.extend(base_agreement)
        round_first_majority_x.append(majority_agreement[0])
        round_mean_majority_x.append(sum(majority_agreement) / depth_count)
        round_y.append(float(accepted_draft_tokens))

        for depth_idx, agreement in enumerate(majority_agreement, start=1):
            token_majority_x.append(agreement)
            if depth_idx - 1 < len(base_agreement):
                token_base_x.append(base_agreement[depth_idx - 1])
            else:
                token_base_x.append(agreement)
            token_y.append(1.0 if depth_idx <= accepted_draft_tokens else 0.0)

    return {
        "rounds": len(round_y),
        "tokens": len(token_y),
        "avg_majority_agreement": sum(all_majority) / len(all_majority) if all_majority else None,
        "avg_base_agreement": sum(all_base) / len(all_base) if all_base else None,
        "avg_accepted_draft_tokens": sum(round_y) / len(round_y) if round_y else None,
        "token_majority_pearson": pearson_correlation(token_majority_x, token_y),
        "token_base_pearson": pearson_correlation(token_base_x, token_y),
        "round_first_majority_pearson": pearson_correlation(round_first_majority_x, round_y),
        "round_mean_majority_pearson": pearson_correlation(round_mean_majority_x, round_y),
    }


def bucket_batch_agreement_metrics(
    metrics: list[dict[str, Any]],
    bucket_edges: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90, 0.95),
) -> list[dict[str, float | int | str]]:
    buckets = [
        {
            "label": f"<{bucket_edges[0]:.2f}",
            "low": float("-inf"),
            "high": bucket_edges[0],
            "count": 0,
            "agreement_sum": 0.0,
            "accepted_sum": 0.0,
        }
    ]
    for low, high in zip(bucket_edges, bucket_edges[1:]):
        buckets.append({
            "label": f"{low:.2f}-{high:.2f}",
            "low": low,
            "high": high,
            "count": 0,
            "agreement_sum": 0.0,
            "accepted_sum": 0.0,
        })
    buckets.append({
        "label": f">={bucket_edges[-1]:.2f}",
        "low": bucket_edges[-1],
        "high": float("inf"),
        "count": 0,
        "agreement_sum": 0.0,
        "accepted_sum": 0.0,
    })

    for metric in metrics:
        majority_agreement = [float(value) for value in metric.get("majority_agreement", [])]
        if not majority_agreement:
            continue

        mean_agreement = sum(majority_agreement) / len(majority_agreement)
        accepted_draft_tokens = float(metric.get("accepted_draft_tokens", 0))
        for bucket in buckets:
            if bucket["low"] <= mean_agreement < bucket["high"]:
                bucket["count"] += 1
                bucket["agreement_sum"] += mean_agreement
                bucket["accepted_sum"] += accepted_draft_tokens
                break

    rows = []
    for bucket in buckets:
        count = int(bucket["count"])
        rows.append({
            "label": str(bucket["label"]),
            "rounds": count,
            "avg_mean_agreement": bucket["agreement_sum"] / count if count else None,
            "avg_accepted_draft_tokens": bucket["accepted_sum"] / count if count else None,
        })
    return rows
