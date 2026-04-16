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
from collections import defaultdict

import torch

from agreement_metrics import bucket_batch_agreement_metrics, summarize_batch_agreement_metrics


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
        for prefix in ("pflash_v2_tb", "pflash_v3_tb", "pflash_v4_tb", "pflash_v5_tb"):
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


def print_batch_agreement_summary(data):
    rows = []
    bucket_rows = []
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
    print_pair_sanity_checks(data)
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
