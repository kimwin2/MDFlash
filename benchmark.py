import argparse
import random
from itertools import chain
from pathlib import Path

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from model import DFlashDraftModel, load_and_process_dataset
from dflash import dflash_generate
from ddtree import ddtree_generate, maybe_enable_cpp_compact
from mdflash import mdflash_generate
from pexpress import pexpress_generate
from pflash import pflash_generate
from pflash_v2 import pflash_v2_generate
from pflash_v3 import pflash_v3_generate
from pflash_v4 import pflash_v4_generate
from pflash_v5 import pflash_v5_generate
from pflash_v6 import pflash_v6_generate
from pflash_v7 import pflash_v7_generate
from exp_ddtree import exp_ddtree_generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tree-budget", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--mdflash-budget", type=str, default=None)
    parser.add_argument("--mdflash-proposal-temperature", type=float, default=1.0)
    parser.add_argument("--pexpress-budget", type=str, default=None)
    parser.add_argument("--pflash-budget", type=str, default=None)
    parser.add_argument("--pflash-v2-budget", type=str, default=None)
    parser.add_argument("--pflash-v3-budget", type=str, default=None)
    parser.add_argument("--pflash-v4-budget", type=str, default=None)
    parser.add_argument("--pflash-v5-budget", type=str, default=None)
    parser.add_argument("--pflash-v6-budget", type=str, default=None)
    parser.add_argument("--pflash-v7-budget", type=str, default=None)
    parser.add_argument("--exp-ddtree-budget", type=str, default=None)
    parser.add_argument("--pexpress-perturbation-temperature", type=float, default=0.75)
    parser.add_argument("--pexpress-position-temperature-decay", type=float, default=0.0)
    parser.add_argument("--pflash-branch-prior-weight", type=float, default=0.5)
    parser.add_argument("--pflash-merge-prefix-branches", action="store_true")
    parser.add_argument("--pflash-prefix-support-bonus-weight", type=float, default=0.0)
    parser.add_argument("--pflash-v4-backbone-fraction", type=float, default=0.75)
    parser.add_argument("--pflash-v4-support-bonus-weight", type=float, default=0.70)
    parser.add_argument("--pflash-v4-base-gap-penalty", type=float, default=0.35)
    parser.add_argument("--pflash-v4-graft-score-threshold", type=float, default=1.0)
    parser.add_argument("--pflash-v5-high-agreement-threshold", type=float, default=0.95)
    parser.add_argument("--pflash-v5-mid-agreement-threshold", type=float, default=0.90)
    parser.add_argument("--pflash-v5-low-agreement-depth", type=int, default=5)
    parser.add_argument("--pflash-v6-high-alignment-threshold", type=float, default=0.95)
    parser.add_argument("--pflash-v6-mid-alignment-threshold", type=float, default=0.90)
    parser.add_argument("--pflash-v6-high-block-size", type=int, default=16)
    parser.add_argument("--pflash-v6-mid-block-size", type=int, default=8)
    parser.add_argument("--pflash-v6-low-block-size", type=int, default=8)
    parser.add_argument("--pflash-v6-high-tree-budget", type=int, default=128)
    parser.add_argument("--pflash-v6-mid-tree-budget", type=int, default=64)
    parser.add_argument("--pflash-v6-low-tree-budget", type=int, default=32)
    parser.add_argument("--measure-batch-agreement", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    visible_cuda_devices = torch.cuda.device_count()
    if visible_cuda_devices <= 0:
        raise RuntimeError("No CUDA devices are visible to benchmark.py.")
    if dist.local_rank() >= visible_cuda_devices:
        raise RuntimeError(
            f"LOCAL_RANK={dist.local_rank()} but only {visible_cuda_devices} CUDA device(s) are visible. "
            "Set NPROC_PER_NODE to the number of visible GPUs."
        )
    if dist.size() > visible_cuda_devices and dist.is_main():
        logger.warning(
            f"WORLD_SIZE={dist.size()} is larger than visible CUDA device count={visible_cuda_devices}. "
            "This usually means multiple benchmark workers are sharing GPUs and the run will be slow."
        )
    if dist.is_main():
        logger.info(
            f"Distributed benchmark config: WORLD_SIZE={dist.size()}, "
            f"LOCAL_WORLD_SIZE={dist.local_size()}, visible_cuda_devices={visible_cuda_devices}"
        )

    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    maybe_enable_cpp_compact(not args.disable_cpp_compact_cache)

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    installed_flash_attn = has_flash_attn()
    if not installed_flash_attn:
        raise RuntimeError("flash_attn must be installed because the draft DFlash model always uses FlashAttention")

    target_attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"
    draft_attn_implementation = "flash_attention_2"

    if not args.flash_attn and installed_flash_attn:
        logger.warning("DDTree, Exp-DDTree, MDFlash, P-Express, P-Flash, P-Flash V2, P-Flash V3, P-Flash V4, P-Flash V5, and P-Flash V6 use a custom tree attention mask on the target model. For compatibility, forcing the target verifier to torch.sdpa.")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=target_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=draft_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tree_budgets = [int(tree_budget) for tree_budget in args.tree_budget.split(",")]
    mdflash_budgets = tree_budgets if args.mdflash_budget is None else [int(tree_budget) for tree_budget in args.mdflash_budget.split(",")]
    pexpress_budgets = tree_budgets if args.pexpress_budget is None else [int(tree_budget) for tree_budget in args.pexpress_budget.split(",")]
    pflash_budgets = tree_budgets if args.pflash_budget is None else [int(tree_budget) for tree_budget in args.pflash_budget.split(",")]
    pflash_v2_budgets = tree_budgets if args.pflash_v2_budget is None else [int(tree_budget) for tree_budget in args.pflash_v2_budget.split(",")]
    pflash_v3_budgets = tree_budgets if args.pflash_v3_budget is None else [int(tree_budget) for tree_budget in args.pflash_v3_budget.split(",")]
    pflash_v4_budgets = tree_budgets if args.pflash_v4_budget is None else [int(tree_budget) for tree_budget in args.pflash_v4_budget.split(",")]
    pflash_v5_budgets = tree_budgets if args.pflash_v5_budget is None else [int(tree_budget) for tree_budget in args.pflash_v5_budget.split(",")]
    pflash_v6_budgets = tree_budgets if args.pflash_v6_budget is None else [int(tree_budget) for tree_budget in args.pflash_v6_budget.split(",")]
    pflash_v7_budgets = [] if args.pflash_v7_budget is None else [int(tree_budget) for tree_budget in args.pflash_v7_budget.split(",")]
    exp_ddtree_budgets = [] if args.exp_ddtree_budget is None else [int(tree_budget) for tree_budget in args.exp_ddtree_budget.split(",")]
    methods_to_run = ["dflash"]
    method_key_to_tree_budget = {}
    if not args.flash_attn:
        mdflash_method_keys = [f"mdflash_tb{tree_budget}" for tree_budget in mdflash_budgets]
        pexpress_method_keys = [f"pexpress_tb{tree_budget}" for tree_budget in pexpress_budgets]
        pflash_method_keys = [f"pflash_tb{tree_budget}" for tree_budget in pflash_budgets]
        pflash_v2_method_keys = [f"pflash_v2_tb{tree_budget}" for tree_budget in pflash_v2_budgets]
        pflash_v3_method_keys = [f"pflash_v3_tb{tree_budget}" for tree_budget in pflash_v3_budgets]
        pflash_v4_method_keys = [f"pflash_v4_tb{tree_budget}" for tree_budget in pflash_v4_budgets]
        pflash_v5_method_keys = [f"pflash_v5_tb{tree_budget}" for tree_budget in pflash_v5_budgets]
        pflash_v6_method_keys = [f"pflash_v6_tb{tree_budget}" for tree_budget in pflash_v6_budgets]
        pflash_v7_method_keys = [f"pflash_v7_tb{tree_budget}" for tree_budget in pflash_v7_budgets]
        exp_ddtree_method_keys = [f"exp_ddtree_tb{tree_budget}" for tree_budget in exp_ddtree_budgets]
        ddtree_method_keys = [f"ddtree_tb{tree_budget}" for tree_budget in tree_budgets]
        methods_to_run.extend(mdflash_method_keys)
        methods_to_run.extend(pexpress_method_keys)
        methods_to_run.extend(pflash_method_keys)
        methods_to_run.extend(pflash_v2_method_keys)
        methods_to_run.extend(pflash_v3_method_keys)
        methods_to_run.extend(pflash_v4_method_keys)
        methods_to_run.extend(pflash_v5_method_keys)
        methods_to_run.extend(pflash_v6_method_keys)
        methods_to_run.extend(pflash_v7_method_keys)
        methods_to_run.extend(exp_ddtree_method_keys)
        methods_to_run.extend(ddtree_method_keys)
        method_key_to_tree_budget.update({f"mdflash_tb{tree_budget}": tree_budget for tree_budget in mdflash_budgets})
        method_key_to_tree_budget.update({f"pexpress_tb{tree_budget}": tree_budget for tree_budget in pexpress_budgets})
        method_key_to_tree_budget.update({f"pflash_tb{tree_budget}": tree_budget for tree_budget in pflash_budgets})
        method_key_to_tree_budget.update({f"pflash_v2_tb{tree_budget}": tree_budget for tree_budget in pflash_v2_budgets})
        method_key_to_tree_budget.update({f"pflash_v3_tb{tree_budget}": tree_budget for tree_budget in pflash_v3_budgets})
        method_key_to_tree_budget.update({f"pflash_v4_tb{tree_budget}": tree_budget for tree_budget in pflash_v4_budgets})
        method_key_to_tree_budget.update({f"pflash_v5_tb{tree_budget}": tree_budget for tree_budget in pflash_v5_budgets})
        method_key_to_tree_budget.update({f"pflash_v6_tb{tree_budget}": tree_budget for tree_budget in pflash_v6_budgets})
        method_key_to_tree_budget.update({f"pflash_v7_tb{tree_budget}": tree_budget for tree_budget in pflash_v7_budgets})
        method_key_to_tree_budget.update({f"exp_ddtree_tb{tree_budget}": tree_budget for tree_budget in exp_ddtree_budgets})
        method_key_to_tree_budget.update({f"ddtree_tb{tree_budget}": tree_budget for tree_budget in tree_budgets})
    else:
        mdflash_method_keys = []
        pexpress_method_keys = []
        pflash_method_keys = []
        pflash_v2_method_keys = []
        pflash_v3_method_keys = []
        pflash_v4_method_keys = []
        pflash_v5_method_keys = []
        pflash_v6_method_keys = []
        pflash_v7_method_keys = []
        exp_ddtree_method_keys = []
        ddtree_method_keys = []

    def run_method(method_key: str, input_ids: torch.Tensor, max_new_tokens: int):
        if method_key == "dflash":
            return dflash_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )

        common_kwargs = {
            "model": draft_model,
            "target": target,
            "input_ids": input_ids,
            "mask_token_id": draft_model.mask_token_id,
            "max_new_tokens": max_new_tokens,
            "block_size": block_size,
            "tree_budget": method_key_to_tree_budget[method_key],
            "stop_token_ids": [tokenizer.eos_token_id],
            "temperature": args.temperature,
        }
        if method_key.startswith("mdflash_tb"):
            return mdflash_generate(
                **common_kwargs,
                proposal_temperature=args.mdflash_proposal_temperature,
            )
        if method_key.startswith("pexpress_tb"):
            return pexpress_generate(
                **common_kwargs,
                perturbation_temperature=args.pexpress_perturbation_temperature,
                position_temperature_decay=args.pexpress_position_temperature_decay,
                measure_batch_agreement=args.measure_batch_agreement,
            )
        if method_key.startswith("pflash_tb"):
            return pflash_generate(
                **common_kwargs,
                perturbation_temperature=args.pexpress_perturbation_temperature,
                position_temperature_decay=args.pexpress_position_temperature_decay,
                branch_prior_weight=args.pflash_branch_prior_weight,
                merge_prefix_branches=args.pflash_merge_prefix_branches,
                prefix_support_bonus_weight=args.pflash_prefix_support_bonus_weight,
                measure_batch_agreement=args.measure_batch_agreement,
            )
        if method_key.startswith("pflash_v2_tb"):
            return pflash_v2_generate(
                **common_kwargs,
                perturbation_temperature=args.pexpress_perturbation_temperature,
                position_temperature_decay=args.pexpress_position_temperature_decay,
                measure_batch_agreement=args.measure_batch_agreement,
            )
        if method_key.startswith("pflash_v3_tb"):
            return pflash_v3_generate(
                **common_kwargs,
                perturbation_temperature=args.pexpress_perturbation_temperature,
                position_temperature_decay=args.pexpress_position_temperature_decay,
                measure_batch_agreement=args.measure_batch_agreement,
            )
        if method_key.startswith("pflash_v4_tb"):
            return pflash_v4_generate(
                **common_kwargs,
                perturbation_temperature=args.pexpress_perturbation_temperature,
                position_temperature_decay=args.pexpress_position_temperature_decay,
                backbone_fraction=args.pflash_v4_backbone_fraction,
                support_bonus_weight=args.pflash_v4_support_bonus_weight,
                base_gap_penalty=args.pflash_v4_base_gap_penalty,
                graft_score_threshold=args.pflash_v4_graft_score_threshold,
                measure_batch_agreement=args.measure_batch_agreement,
            )
        if method_key.startswith("pflash_v5_tb"):
            return pflash_v5_generate(
                **common_kwargs,
                perturbation_temperature=args.pexpress_perturbation_temperature,
                position_temperature_decay=args.pexpress_position_temperature_decay,
                high_agreement_threshold=args.pflash_v5_high_agreement_threshold,
                mid_agreement_threshold=args.pflash_v5_mid_agreement_threshold,
                low_agreement_depth=args.pflash_v5_low_agreement_depth,
                measure_batch_agreement=args.measure_batch_agreement,
            )
        if method_key.startswith("pflash_v6_tb"):
            return pflash_v6_generate(
                **common_kwargs,
                perturbation_temperature=args.pexpress_perturbation_temperature,
                position_temperature_decay=args.pexpress_position_temperature_decay,
                high_alignment_threshold=args.pflash_v6_high_alignment_threshold,
                mid_alignment_threshold=args.pflash_v6_mid_alignment_threshold,
                high_block_size=args.pflash_v6_high_block_size,
                mid_block_size=args.pflash_v6_mid_block_size,
                low_block_size=args.pflash_v6_low_block_size,
                high_tree_budget=args.pflash_v6_high_tree_budget,
                mid_tree_budget=args.pflash_v6_mid_tree_budget,
                low_tree_budget=args.pflash_v6_low_tree_budget,
                measure_batch_agreement=args.measure_batch_agreement,
            )
        if method_key.startswith("pflash_v7_tb"):
            return pflash_v7_generate(
                **common_kwargs,
                measure_batch_agreement=args.measure_batch_agreement,
            )
        if method_key.startswith("exp_ddtree_tb"):
            return exp_ddtree_generate(
                **common_kwargs,
                perturbation_temperature=args.pexpress_perturbation_temperature,
                position_temperature_decay=args.pexpress_position_temperature_decay,
            )
        if method_key.startswith("ddtree_tb"):
            return ddtree_generate(**common_kwargs)
        raise ValueError(f"Unsupported method key: {method_key}")

    if ddtree_method_keys:
        history_method_key = ddtree_method_keys[-1]
    elif exp_ddtree_method_keys:
        history_method_key = exp_ddtree_method_keys[-1]
    elif pflash_method_keys:
        history_method_key = pflash_method_keys[-1]
    elif pflash_v2_method_keys:
        history_method_key = pflash_v2_method_keys[-1]
    elif pflash_v3_method_keys:
        history_method_key = pflash_v3_method_keys[-1]
    elif pflash_v4_method_keys:
        history_method_key = pflash_v4_method_keys[-1]
    elif pflash_v5_method_keys:
        history_method_key = pflash_v5_method_keys[-1]
    elif pflash_v6_method_keys:
        history_method_key = pflash_v6_method_keys[-1]
    elif pflash_v7_method_keys:
        history_method_key = pflash_v7_method_keys[-1]
    elif pexpress_method_keys:
        history_method_key = pexpress_method_keys[-1]
    elif mdflash_method_keys:
        history_method_key = mdflash_method_keys[-1]
    else:
        history_method_key = "dflash"

    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(warmup_input_text, return_tensors="pt").to(target.device)
    warmup_max_new_tokens = min(args.max_new_tokens, 16)

    _ = dflash_generate(
        model=draft_model,
        target=target,
        input_ids=warmup_input_ids,
        mask_token_id=draft_model.mask_token_id,
        max_new_tokens=warmup_max_new_tokens,
        block_size=1,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
    )
    for method_key in methods_to_run:
        _ = run_method(method_key, warmup_input_ids, warmup_max_new_tokens)

    responses = []
    response_metadata = []
    indices = list(range(dist.rank(), len(dataset), dist.size()))
    logger.info(
        f"Rank {dist.rank()}/{dist.size()} assigned {len(indices)} sample(s): "
        f"{indices[:16]}{'...' if len(indices) > 16 else ''}"
    )
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        turns = list(instance["turns"])
        for turn_idx, user_content in enumerate(turns):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            response["baseline"] = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=1,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
            for method_key in methods_to_run:
                response[method_key] = run_method(method_key, input_ids, args.max_new_tokens)

            spec_response = response[history_method_key]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)
            response_metadata.append({
                "dataset_index": int(idx),
                "turn_index": int(turn_idx),
                "num_turns": int(len(turns)),
                "rank": int(dist.rank()),
            })

    if dist.size() > 1:
        gathered = dist.gather({
            "responses": responses,
            "response_metadata": response_metadata,
        }, dst=0)
        if not dist.is_main():
            return
        responses = list(chain.from_iterable(shard["responses"] for shard in gathered))
        response_metadata = list(chain.from_iterable(shard["response_metadata"] for shard in gathered))

    response_order = sorted(
        range(len(response_metadata)),
        key=lambda index: (
            response_metadata[index]["dataset_index"],
            response_metadata[index]["turn_index"],
            response_metadata[index]["rank"],
        ),
    )
    responses = [responses[index] for index in response_order]
    response_metadata = [response_metadata[index] for index in response_order]

    run_data = {
        "responses": responses,
        "response_metadata": response_metadata,
        "block_size": block_size,
        "draft_attn_implementation": draft_attn_implementation,
        "target_attn_implementation": target_attn_implementation,
        "args": vars(args),
    }
    
    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run_data, save_path)


if __name__ == "__main__":
    main()
