from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from agreement_metrics import append_batch_agreement_metric
from dflash import cuda_time, dflash_generate, empty_stage_times
from model import DFlashDraftModel, extract_context_feature, sample
from pflash_v2 import repeat_dynamic_cache_batch, select_dynamic_cache_batch


PFLASH_V7_BATCH_SIZE = 4
PFLASH_V7_STAGE_ORDER = ("draft", "candidate_sample", "verify", "commit")


def select_multiverse_anchor_tokens(
    anchor_logits: torch.Tensor,
    branch0_token_id: int,
    num_branches: int,
) -> tuple[torch.Tensor, list[int | None]]:
    logits = anchor_logits.float()
    topk_count = min(int(logits.shape[-1]), max(num_branches * 8, num_branches))
    ranked_tokens = torch.topk(logits, k=topk_count, dim=-1).indices.tolist()
    rank_by_token = {int(token_id): rank + 1 for rank, token_id in enumerate(ranked_tokens)}

    selected_tokens = [int(branch0_token_id)]
    for token_id in ranked_tokens:
        token_id = int(token_id)
        if token_id not in selected_tokens:
            selected_tokens.append(token_id)
        if len(selected_tokens) >= num_branches:
            break

    while len(selected_tokens) < num_branches:
        selected_tokens.append(int(branch0_token_id))

    anchor_ranks = [rank_by_token.get(token_id) for token_id in selected_tokens]
    return (
        torch.tensor(selected_tokens, dtype=torch.long, device=anchor_logits.device),
        anchor_ranks,
    )


def select_best_linear_branch(
    verify_input_ids: torch.Tensor,
    posterior: torch.Tensor,
) -> tuple[int, list[int], int, list[int]]:
    branch_acceptance_lengths = []
    best_branch_idx = 0
    best_accepted_length = 1

    for branch_idx in range(verify_input_ids.shape[0]):
        accepted_draft_tokens = int(
            (verify_input_ids[branch_idx : branch_idx + 1, 1:] == posterior[branch_idx : branch_idx + 1, :-1])
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        accepted_length = 1 + accepted_draft_tokens
        branch_acceptance_lengths.append(int(accepted_length))
        if accepted_length > best_accepted_length:
            best_branch_idx = branch_idx
            best_accepted_length = accepted_length

    accepted_indices = list(range(best_accepted_length))
    next_token = int(posterior[best_branch_idx, best_accepted_length - 1])
    return best_branch_idx, accepted_indices, next_token, branch_acceptance_lengths


@torch.inference_mode()
def pflash_v7_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    measure_batch_agreement: bool = False,
    save_tree_traces: bool = False,
) -> SimpleNamespace:
    if block_size <= 1:
        return dflash_generate(
            model=model,
            target=target,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_input_ids_buffer = torch.full(
        (PFLASH_V7_BATCH_SIZE, block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    verify_position_ids_buffer = position_ids[:, :block_size].expand(PFLASH_V7_BATCH_SIZE, -1).clone()

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(PFLASH_V7_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    initial_root_token = sample(output.logits, temperature)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = initial_root_token
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    anchor_logits = output.logits[0, 0].detach()

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    round_trees = [] if save_tree_traces else None
    batch_agreement_metrics = [] if measure_batch_agreement else None
    pflash_v7_metrics: list[dict[str, Any]] = []
    draft_prefill = True

    while start < max_length:
        base_root_token = int(output_ids[0, start])
        anchor_tokens, anchor_ranks = select_multiverse_anchor_tokens(
            anchor_logits=anchor_logits,
            branch0_token_id=base_root_token,
            num_branches=PFLASH_V7_BATCH_SIZE,
        )

        draft_stage_start = cuda_time()
        # Experimental multiverse rollout: branch 0 keeps the committed anchor
        # token, while the remaining branches swap in alternate top-k anchors.
        branch_block_output_ids = verify_input_ids_buffer
        branch_block_output_ids.fill_(mask_token_id)
        branch_block_output_ids[:, 0].copy_(anchor_tokens)

        draft_past_key_values = repeat_dynamic_cache_batch(
            past_key_values=past_key_values_draft,
            batch_size=PFLASH_V7_BATCH_SIZE,
        )
        projected_target_hidden = model.project_target_hidden(target_hidden)
        batched_target_hidden = projected_target_hidden.expand(PFLASH_V7_BATCH_SIZE, -1, -1)
        draft_position_ids = position_ids[
            :,
            past_key_values_draft.get_seq_length() : start + block_size,
        ].expand(PFLASH_V7_BATCH_SIZE, -1)
        draft_logits = target.lm_head(model(
            target_hidden=batched_target_hidden,
            target_hidden_is_projected=True,
            noise_embedding=target.model.embed_tokens(branch_block_output_ids),
            position_ids=draft_position_ids,
            past_key_values=draft_past_key_values,
            use_cache=True,
            is_causal=False,
        )[:, -draft_horizon:, :])
        past_key_values_draft = select_dynamic_cache_batch(draft_past_key_values, 0)
        past_key_values_draft.crop(start)
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        sample_stage_start = cuda_time()
        candidate_token_ids = torch.argmax(draft_logits, dim=-1)
        if draft_horizon > 0:
            branch_block_output_ids[:, 1:] = candidate_token_ids
        verify_position_ids = verify_position_ids_buffer[:, :block_size]
        verify_position_ids.copy_(position_ids[:, start : start + block_size].expand(PFLASH_V7_BATCH_SIZE, -1))
        stage_times["candidate_sample"] += cuda_time() - sample_stage_start

        verify_stage_start = cuda_time()
        verify_past_key_values = repeat_dynamic_cache_batch(
            past_key_values=past_key_values_target,
            batch_size=PFLASH_V7_BATCH_SIZE,
        )
        output = target(
            branch_block_output_ids,
            position_ids=verify_position_ids,
            past_key_values=verify_past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        selected_branch_idx, accepted_indices, next_token, branch_acceptance_lengths = select_best_linear_branch(
            verify_input_ids=branch_block_output_ids,
            posterior=posterior,
        )
        append_batch_agreement_metric(batch_agreement_metrics, draft_logits, accepted_indices)
        accepted_length = len(accepted_indices)

        selected_tokens = branch_block_output_ids[selected_branch_idx : selected_branch_idx + 1, :accepted_length]
        output_ids[:, start : start + accepted_length] = selected_tokens
        output_ids[:, start + accepted_length] = next_token

        past_key_values_target = select_dynamic_cache_batch(verify_past_key_values, selected_branch_idx)
        past_key_values_target.crop(start + accepted_length)

        target_hidden_batch = extract_context_feature(output.hidden_states, model.target_layer_ids)
        target_hidden = target_hidden_batch[selected_branch_idx : selected_branch_idx + 1, :accepted_length, :]
        anchor_logits = output.logits[selected_branch_idx, accepted_length - 1].detach()

        pflash_v7_metrics.append({
            "anchor_tokens": [int(token_id) for token_id in anchor_tokens.tolist()],
            "anchor_ranks": [None if rank is None else int(rank) for rank in anchor_ranks],
            "branch_acceptance_lengths": [int(length) for length in branch_acceptance_lengths],
            "selected_branch": int(selected_branch_idx),
            "selected_anchor_rank": (
                None if anchor_ranks[selected_branch_idx] is None else int(anchor_ranks[selected_branch_idx])
            ),
            "base_acceptance_length": int(branch_acceptance_lengths[0]),
            "selected_acceptance_length": int(branch_acceptance_lengths[selected_branch_idx]),
            "alternative_branch_selected": bool(selected_branch_idx != 0),
        })

        acceptance_lengths.append(accepted_length)
        start += accepted_length
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if save_tree_traces:
            round_trees.append({
                "selected_branch": int(selected_branch_idx),
                "anchor_tokens": [int(token_id) for token_id in anchor_tokens.tolist()],
                "branch_acceptance_lengths": [int(length) for length in branch_acceptance_lengths],
                "candidate_token_ids": [
                    [int(token_id) for token_id in branch_tokens]
                    for branch_tokens in candidate_token_ids.tolist()
                ],
            })

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - accepted_length : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        round_trees=round_trees,
        batch_agreement_metrics=batch_agreement_metrics,
        pflash_v7_metrics=pflash_v7_metrics,
    )
