#!/usr/bin/env bash

set -euo pipefail

DIR="${1:-runs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"
export GETLOG_DIR="${DIR}"

python3 <<'PY'
import glob
import os
import sys
from collections import Counter, defaultdict

import torch

from agreement_metrics import (
    bucket_batch_agreement_metrics,
    pearson_correlation,
    summarize_batch_agreement_metrics,
)


def load_and_analyze(pt_path):
    try:
        data = torch.load(pt_path, weights_only=False, map_location="cpu")
    except Exception as exc:
        return None, f"Error loading {pt_path}: {exc}"

    responses = data.get("responses", [])
    if not responses:
        return None, f"No responses found in {pt_path}"

    first_response = responses[0]
    methods = list(first_response.keys())

    results = defaultdict(list)
    for response in responses:
        for method in methods:
            if method not in response:
                continue
            result = response[method]
            time_per_output_token = float(result.time_per_output_token)
            num_output_tokens = int(result.num_output_tokens)
            total_time = num_output_tokens * time_per_output_token
            acceptance_lengths = list(result.acceptance_lengths)
            results[method].append({
                "num_output_tokens": num_output_tokens,
                "time_per_output_token": time_per_output_token,
                "total_time": total_time,
                "acceptance_lengths": acceptance_lengths,
                "decode_rounds": int(result.decode_rounds),
            })

    return {
        "methods": methods,
        "responses": responses,
        "response_metadata": data.get("response_metadata", []),
        "results": dict(results),
        "args": data.get("args", {}),
        "block_size": data.get("block_size", "N/A"),
        "target_attn": data.get("target_attn_implementation", "N/A"),
    }, None


def summarize_method(rows):
    if not rows:
        return {
            "avg_tokens": 0.0,
            "tps": 0.0,
            "avg_acceptance_length": 0.0,
            "total_time": 0.0,
            "avg_rounds": 0.0,
        }

    total_tokens = sum(row["num_output_tokens"] for row in rows)
    total_time = sum(row["total_time"] for row in rows)
    all_acceptance_lengths = [
        length
        for row in rows
        for length in row["acceptance_lengths"]
    ]
    return {
        "avg_tokens": total_tokens / len(rows),
        "tps": total_tokens / total_time if total_time > 0 else 0.0,
        "avg_acceptance_length": (
            sum(all_acceptance_lengths) / len(all_acceptance_lengths)
            if all_acceptance_lengths
            else 0.0
        ),
        "total_time": total_time,
        "avg_rounds": sum(row["decode_rounds"] for row in rows) / len(rows),
    }


def same_tensor(lhs, rhs):
    if not hasattr(lhs, "shape") or not hasattr(rhs, "shape"):
        return lhs == rhs
    return torch.equal(lhs, rhs)


def print_pair_sanity_checks(data):
    methods = set(data["methods"])
    responses = data["responses"]
    checked_pairs = []

    for method in sorted(methods):
        for prefix in ("exp_ddtree_tb", "pflash_v2_tb", "pflash_v3_tb", "pflash_v4_tb", "pflash_v5_tb", "pflash_v6_tb"):
            if not method.startswith(prefix):
                continue
            budget = method.removeprefix(prefix)
            ddtree_method = f"ddtree_tb{budget}"
            if ddtree_method in methods:
                checked_pairs.append((method, ddtree_method))

    if not checked_pairs:
        return

    print("-" * 120)
    print("Sanity checks against DDTree")
    for left_method, right_method in checked_pairs:
        same_object = 0
        same_output_ids = 0
        same_acceptance = 0
        same_stage_times = 0
        same_round_timestamps = 0

        for response in responses:
            left = response[left_method]
            right = response[right_method]
            same_object += int(left is right)
            same_output_ids += int(same_tensor(left.output_ids, right.output_ids))
            same_acceptance += int(left.acceptance_lengths == right.acceptance_lengths)
            same_stage_times += int(left.stage_times == right.stage_times)
            same_round_timestamps += int(left.round_timestamps == right.round_timestamps)

        count = len(responses)
        print(
            f"  {left_method} vs {right_method}: "
            f"object={same_object}/{count}, "
            f"output={same_output_ids}/{count}, "
            f"acc={same_acceptance}/{count}, "
            f"stage={same_stage_times}/{count}, "
            f"round_ts={same_round_timestamps}/{count}"
        )


def print_sample_coverage(data):
    metadata = data.get("response_metadata") or []
    responses = data["responses"]
    if not metadata:
        print("-" * 120)
        print("Sample coverage metadata: unavailable. Re-run benchmark.py after the metadata patch to check duplicates.")
        return

    print("-" * 120)
    print("Sample coverage")
    if len(metadata) != len(responses):
        print(f"  WARNING: metadata count {len(metadata)} != response count {len(responses)}")

    keys = [
        (int(item.get("dataset_index", -1)), int(item.get("turn_index", -1)))
        for item in metadata
    ]
    key_counts = Counter(keys)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    rank_counts = Counter(int(item.get("rank", -1)) for item in metadata)
    sample_counts = Counter(int(item.get("dataset_index", -1)) for item in metadata)

    print(f"  response_entries={len(responses)}, unique_sample_turns={len(key_counts)}, duplicate_sample_turns={len(duplicate_keys)}")
    print("  rank_counts: " + ", ".join(f"rank{rank}={count}" for rank, count in sorted(rank_counts.items())))
    print(f"  unique_samples={len(sample_counts)}, sample_turn_count_range={min(sample_counts.values())}-{max(sample_counts.values())}")
    if duplicate_keys:
        preview = duplicate_keys[:12]
        print("  duplicate keys preview: " + ", ".join(f"sample{idx}/turn{turn}" for idx, turn in preview))


def summarize_adaptive_modes(metrics):
    by_mode = {}
    for metric in metrics:
        mode = metric.get("adaptive_mode")
        if mode is None:
            continue

        majority_agreement = [float(value) for value in metric.get("majority_agreement", [])]
        if not majority_agreement:
            continue

        row = by_mode.setdefault(mode, {
            "mode": mode,
            "rounds": 0,
            "block_sum": 0.0,
            "tree_sum": 0.0,
            "agreement_sum": 0.0,
            "accepted_sum": 0.0,
        })
        row["rounds"] += 1
        row["block_sum"] += float(metric.get("effective_block_size", 0))
        row["tree_sum"] += float(metric.get("effective_tree_budget", 0))
        row["agreement_sum"] += sum(majority_agreement) / len(majority_agreement)
        row["accepted_sum"] += float(metric.get("accepted_draft_tokens", 0))

    mode_order = {"high": 0, "mid": 1, "low": 2}
    rows = []
    for row in by_mode.values():
        rounds = row["rounds"]
        rows.append({
            "mode": row["mode"],
            "rounds": rounds,
            "avg_block_size": row["block_sum"] / rounds,
            "avg_tree_budget": row["tree_sum"] / rounds,
            "avg_mean_agreement": row["agreement_sum"] / rounds,
            "avg_accepted_draft_tokens": row["accepted_sum"] / rounds,
        })
    return sorted(rows, key=lambda row: mode_order.get(row["mode"], 100))


def summarize_exp_ddtree_metrics(metrics):
    if not metrics:
        return None

    accepted = [float(metric.get("accepted_draft_tokens", 0.0)) for metric in metrics]
    mean_alignment = [float(metric.get("mean_alignment", 0.0)) for metric in metrics]
    max_depth = [float(metric.get("tree_max_depth", 0.0)) for metric in metrics]
    max_width = [float(metric.get("tree_max_width", 0.0)) for metric in metrics]
    mean_abs_diff = [float(metric.get("logit_mean_abs_diff", 0.0)) for metric in metrics]
    max_abs_diff = [float(metric.get("logit_max_abs_diff", 0.0)) for metric in metrics]
    top1_match_rate = [float(metric.get("top1_match_rate", 0.0)) for metric in metrics]
    rounds_with_top1_drift = sum(0 if metric.get("top1_all_match", False) else 1 for metric in metrics)

    return {
        "rounds": len(metrics),
        "align_acc_pearson": pearson_correlation(mean_alignment, accepted),
        "depth_acc_pearson": pearson_correlation(max_depth, accepted),
        "width_acc_pearson": pearson_correlation(max_width, accepted),
        "align_depth_pearson": pearson_correlation(mean_alignment, max_depth),
        "align_width_pearson": pearson_correlation(mean_alignment, max_width),
        "drift_acc_pearson": pearson_correlation(mean_abs_diff, accepted),
        "avg_alignment": sum(mean_alignment) / len(mean_alignment) if mean_alignment else None,
        "avg_depth": sum(max_depth) / len(max_depth) if max_depth else None,
        "avg_width": sum(max_width) / len(max_width) if max_width else None,
        "avg_acc": sum(accepted) / len(accepted) if accepted else None,
        "avg_logit_mean_abs_diff": sum(mean_abs_diff) / len(mean_abs_diff) if mean_abs_diff else None,
        "avg_logit_max_abs_diff": sum(max_abs_diff) / len(max_abs_diff) if max_abs_diff else None,
        "avg_top1_match_rate": sum(top1_match_rate) / len(top1_match_rate) if top1_match_rate else None,
        "rounds_with_top1_drift": int(rounds_with_top1_drift),
    }


def print_exp_ddtree_summary(data):
    rows = []
    responses = data["responses"]
    for method in data["methods"]:
        collected_metrics = []
        for response in responses:
            result = response.get(method)
            if result is None:
                continue
            collected_metrics.extend(getattr(result, "exp_ddtree_metrics", None) or [])
        summary = summarize_exp_ddtree_metrics(collected_metrics)
        if summary is not None:
            rows.append((method, summary))

    if not rows:
        return

    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"

    print("-" * 120)
    print("Exp-DDTree shape vs alignment vs acceptance")
    print(
        "{:<20} | {:>7} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>8} | {:>8} | {:>8}".format(
            "Method",
            "Rounds",
            "Aln-Acc",
            "Dep-Acc",
            "Wid-Acc",
            "Aln-Dep",
            "Aln-Wid",
            "AvgAln",
            "AvgDep",
            "AvgAcc",
        )
    )
    print("-" * 120)
    for method, summary in rows:
        print(
            "{:<20} | {:>7} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>8} | {:>8} | {:>8}".format(
                method,
                summary["rounds"],
                fmt(summary["align_acc_pearson"]),
                fmt(summary["depth_acc_pearson"]),
                fmt(summary["width_acc_pearson"]),
                fmt(summary["align_depth_pearson"]),
                fmt(summary["align_width_pearson"]),
                fmt(summary["avg_alignment"]),
                fmt(summary["avg_depth"]),
                fmt(summary["avg_acc"]),
            )
        )
    print("  Pearson r is computed round-wise over accepted draft tokens.")

    print("-" * 120)
    print("Exp-DDTree single-vs-batch0 draft drift")
    print(
        "{:<20} | {:>7} | {:>10} | {:>10} | {:>8} | {:>10} | {:>9}".format(
            "Method",
            "Rounds",
            "MeanAbs",
            "MaxAbs",
            "Top1Eq",
            "DriftRnd",
            "DriftAcc",
        )
    )
    print("-" * 120)
    for method, summary in rows:
        print(
            "{:<20} | {:>7} | {:>10} | {:>10} | {:>8} | {:>10} | {:>9}".format(
                method,
                summary["rounds"],
                fmt(summary["avg_logit_mean_abs_diff"]),
                fmt(summary["avg_logit_max_abs_diff"]),
                fmt(summary["avg_top1_match_rate"]),
                summary["rounds_with_top1_drift"],
                fmt(summary["drift_acc_pearson"]),
            )
        )
    print("  MeanAbs/MaxAbs compare the single-branch DDTree draft logits against batch-0 logits from the batched alignment pass.")


def summarize_pflash_v7_metrics(metrics):
    if not metrics:
        return None

    rounds = len(metrics)
    branch_win_counts = Counter(int(metric.get("selected_branch", -1)) for metric in metrics)
    alt_branch_selected = sum(1 for metric in metrics if bool(metric.get("alternative_branch_selected", False)))
    base_acceptance = [float(metric.get("base_acceptance_length", 0.0)) for metric in metrics]
    selected_acceptance = [float(metric.get("selected_acceptance_length", 0.0)) for metric in metrics]
    selected_ranks = [
        float(metric["selected_anchor_rank"])
        for metric in metrics
        if metric.get("selected_anchor_rank") is not None
    ]

    return {
        "rounds": rounds,
        "alt_win_rate": alt_branch_selected / rounds if rounds else None,
        "avg_base_acceptance": sum(base_acceptance) / rounds if rounds else None,
        "avg_selected_acceptance": sum(selected_acceptance) / rounds if rounds else None,
        "avg_gain": (sum(selected_acceptance) - sum(base_acceptance)) / rounds if rounds else None,
        "avg_selected_rank": sum(selected_ranks) / len(selected_ranks) if selected_ranks else None,
        "branch_win_counts": [int(branch_win_counts.get(branch_idx, 0)) for branch_idx in range(4)],
    }


def print_pflash_v7_summary(data):
    rows = []
    responses = data["responses"]
    for method in data["methods"]:
        collected_metrics = []
        for response in responses:
            result = response.get(method)
            if result is None:
                continue
            collected_metrics.extend(getattr(result, "pflash_v7_metrics", None) or [])
        summary = summarize_pflash_v7_metrics(collected_metrics)
        if summary is not None:
            rows.append((method, summary))

    if not rows:
        return

    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"

    print("-" * 120)
    print("P-Flash V7 multiverse branch routing")
    print(
        "{:<20} | {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>6} | {:>6} | {:>6} | {:>6}".format(
            "Method",
            "Rounds",
            "AltWin",
            "BaseAcc",
            "BestAcc",
            "Gain",
            "SelRank",
            "B0",
            "B1",
            "B2",
            "B3",
        )
    )
    print("-" * 120)
    for method, summary in rows:
        print(
            "{:<20} | {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>6} | {:>6} | {:>6} | {:>6}".format(
                method,
                summary["rounds"],
                fmt(summary["alt_win_rate"]),
                fmt(summary["avg_base_acceptance"]),
                fmt(summary["avg_selected_acceptance"]),
                fmt(summary["avg_gain"]),
                fmt(summary["avg_selected_rank"]),
                summary["branch_win_counts"][0],
                summary["branch_win_counts"][1],
                summary["branch_win_counts"][2],
                summary["branch_win_counts"][3],
            )
        )
    print("  AltWin is the fraction of rounds where a non-base anchor branch beat branch 0.")


def print_batch_agreement_summary(data):
    rows = []
    bucket_rows = []
    adaptive_rows = []
    responses = data["responses"]
    for method in data["methods"]:
        collected_metrics = []
        for response in responses:
            result = response.get(method)
            if result is None:
                continue
            collected_metrics.extend(getattr(result, "batch_agreement_metrics", None) or [])
        if not collected_metrics:
            continue
        summary = summarize_batch_agreement_metrics(collected_metrics)
        if summary["rounds"] == 0:
            continue
        rows.append((method, summary))
        bucket_rows.extend(
            (method, bucket)
            for bucket in bucket_batch_agreement_metrics(collected_metrics)
            if bucket["rounds"] > 0
        )
        adaptive_rows.extend((method, row) for row in summarize_adaptive_modes(collected_metrics))

    if not rows:
        return

    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"

    print("-" * 120)
    print("Batch agreement vs actual acceptance")
    print(
        "{:<20} | {:>7} | {:>8} | {:>9} | {:>9} | {:>10} | {:>10} | {:>8} | {:>8}".format(
            "Method",
            "Rounds",
            "Tokens",
            "TokMaj r",
            "TokBase r",
            "RndFirst r",
            "RndMean r",
            "AvgMaj",
            "AvgAcc",
        )
    )
    print("-" * 120)
    for method, summary in rows:
        print(
            "{:<20} | {:>7} | {:>8} | {:>9} | {:>9} | {:>10} | {:>10} | {:>8} | {:>8}".format(
                method,
                summary["rounds"],
                summary["tokens"],
                fmt(summary["token_majority_pearson"]),
                fmt(summary["token_base_pearson"]),
                fmt(summary["round_first_majority_pearson"]),
                fmt(summary["round_mean_majority_pearson"]),
                fmt(summary["avg_majority_agreement"]),
                fmt(summary["avg_accepted_draft_tokens"]),
            )
        )
    print("  Tok* r: token-level Pearson against per-depth accepted/not-accepted.")
    print("  Rnd* r: round-level Pearson against accepted draft-token count.")

    if not bucket_rows:
        return

    print("-" * 120)
    print("Round mean agreement buckets")
    print(
        "{:<20} | {:>11} | {:>7} | {:>8} | {:>8}".format(
            "Method",
            "MeanAgr",
            "Rounds",
            "AvgMean",
            "AvgAcc",
        )
    )
    print("-" * 120)
    for method, bucket in bucket_rows:
        print(
            "{:<20} | {:>11} | {:>7} | {:>8} | {:>8}".format(
                method,
                bucket["label"],
                bucket["rounds"],
                fmt(bucket["avg_mean_agreement"]),
                fmt(bucket["avg_accepted_draft_tokens"]),
            )
        )

    if adaptive_rows:
        print("-" * 120)
        print("Adaptive alignment routing")
        print(
            "{:<20} | {:>8} | {:>7} | {:>8} | {:>8} | {:>8} | {:>8}".format(
                "Method",
                "Mode",
                "Rounds",
                "AvgBlk",
                "AvgTree",
                "AvgAgr",
                "AvgAcc",
            )
        )
        print("-" * 120)
        for method, row in adaptive_rows:
            print(
                "{:<20} | {:>8} | {:>7} | {:>8} | {:>8} | {:>8} | {:>8}".format(
                    method,
                    row["mode"],
                    row["rounds"],
                    fmt(row["avg_block_size"]),
                    fmt(row["avg_tree_budget"]),
                    fmt(row["avg_mean_agreement"]),
                    fmt(row["avg_accepted_draft_tokens"]),
                )
            )


def print_single_result(data, filename):
    methods = data["methods"]
    results = data["results"]
    args = data["args"]

    print("=" * 120)
    print(f"FILE: {filename}")
    print("=" * 120)
    print("Block size: {}, Target attn: {}".format(data["block_size"], data["target_attn"]))
    print("Args:")
    for name in (
        "tree_budget",
        "pflash_budget",
        "pflash_v2_budget",
        "pflash_v3_budget",
        "pflash_v4_budget",
        "pflash_v5_budget",
        "pflash_v6_budget",
        "pflash_v7_budget",
        "exp_ddtree_budget",
        "pexpress_perturbation_temperature",
        "pexpress_position_temperature_decay",
        "pflash_branch_prior_weight",
        "pflash_merge_prefix_branches",
        "pflash_prefix_support_bonus_weight",
        "pflash_v4_backbone_fraction",
        "pflash_v4_support_bonus_weight",
        "pflash_v4_base_gap_penalty",
        "pflash_v4_graft_score_threshold",
        "pflash_v5_high_agreement_threshold",
        "pflash_v5_mid_agreement_threshold",
        "pflash_v5_low_agreement_depth",
        "pflash_v6_high_alignment_threshold",
        "pflash_v6_mid_alignment_threshold",
        "pflash_v6_high_block_size",
        "pflash_v6_mid_block_size",
        "pflash_v6_low_block_size",
        "pflash_v6_high_tree_budget",
        "pflash_v6_mid_tree_budget",
        "pflash_v6_low_tree_budget",
        "measure_batch_agreement",
    ):
        print("  {}={}".format(name, args.get(name, "N/A")))
    print("-" * 120)

    header = "{:<20} | {:>10} | {:>8} | {:>10} | {:>12} | {:>10}".format(
        "Method", "Avg Tokens", "TPS", "Avg Acc Len", "Total Time (s)", "Rounds"
    )
    print(header)
    print("-" * 120)

    for method in methods:
        if not results.get(method):
            continue
        summary = summarize_method(results[method])
        print("{:<20} | {:>10.1f} | {:>8.1f} | {:>10.2f} | {:>12.2f} | {:>10.1f}".format(
            method,
            summary["avg_tokens"],
            summary["tps"],
            summary["avg_acceptance_length"],
            summary["total_time"],
            summary["avg_rounds"],
        ))

    print_batch_agreement_summary(data)
    print_exp_ddtree_summary(data)
    print_pflash_v7_summary(data)
    print_pair_sanity_checks(data)
    print_sample_coverage(data)
    print()


def compare_results(all_data):
    print("\\n" + "=" * 140)
    print("COMPARISON SUMMARY (All Files)")
    print("=" * 140)

    all_methods = set()
    for data, _ in all_data:
        all_methods.update(data["methods"])
    all_methods = sorted(all_methods)

    header = "{:<50} | ".format("File")
    for method in all_methods:
        if method != "baseline":
            header += "{:>15} | ".format(method[:15])
    print(header)
    print("-" * 140)

    for data, filename in all_data:
        results = data["results"]
        row = "{:<50} | ".format(filename[:50])
        for method in all_methods:
            if method == "baseline":
                continue
            if results.get(method):
                summary = summarize_method(results[method])
                row += "{:>15.1f} | ".format(summary["tps"])
            else:
                row += "{:>15} | ".format("-")
        print(row)

    print("=" * 140)

    print("\\nSPEEDUP vs BASELINE:")
    print("-" * 140)
    header = "{:<50} | ".format("File")
    for method in all_methods:
        if method != "baseline":
            header += "{:>15} | ".format(method[:15])
    print(header)
    print("-" * 140)

    for data, filename in all_data:
        results = data["results"]
        if not results.get("baseline"):
            continue

        baseline_tps = summarize_method(results["baseline"])["tps"]
        row = "{:<50} | ".format(filename[:50])
        for method in all_methods:
            if method == "baseline":
                continue
            if results.get(method):
                avg_tps = summarize_method(results[method])["tps"]
                speedup = avg_tps / baseline_tps if baseline_tps > 0 else 0.0
                row += "{:>14.2f}x | ".format(speedup)
            else:
                row += "{:>15} | ".format("-")
        print(row)

    print("=" * 140)


dir_path = os.environ["GETLOG_DIR"]
pt_files = sorted(glob.glob(os.path.join(dir_path, "*.pt")))

if not pt_files:
    print(f"No .pt files found in {dir_path}")
    sys.exit(1)

print(f"Found {len(pt_files)} .pt files in {dir_path}")
print()

all_data = []
for pt_path in pt_files:
    filename = os.path.basename(pt_path)
    data, error = load_and_analyze(pt_path)
    if error:
        print(error)
        continue
    all_data.append((data, filename))
    print_single_result(data, filename)

if len(all_data) > 1:
    compare_results(all_data)
PY
