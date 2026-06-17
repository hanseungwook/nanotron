"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```
"""

import argparse
import glob
import json
import time
from pprint import pformat
from typing import Dict, Optional, cast

from torch.utils.data import DataLoader

import nanotron.distributed as dist
from nanotron import logging
from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
    NanosetDatasetsArgs,
    PretrainDatasetsArgs,
    Qwen2Config,
    SFTDatasetsArgs,
)
from nanotron.data.dataloader import (
    dummy_infinite_data_generator,
    get_train_dataloader,
)
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.data.processing import (
    clm_process,
    get_datasets,
)
from nanotron.data.sft_processing import prepare_sft_dataset
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.sanity_checks import sanity_check_dataloader
from nanotron.trainer import DistributedTrainer
from nanotron.utils import main_rank_first

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)


# ---------------------------
# Diagnostic logger (added)
# ---------------------------
def _log_stage_start_plan(
    trainer: "DistributedTrainer",
    stage: DatasetStageArgs,
    consumed_tokens_per_dataset_folder: Dict[str, int],
    remaining_train_steps: int,
) -> None:
    """
    Pretty, localized stage-start summary. Prints folder-level offsets and the
    expected sampling plan for the upcoming stage window.

    - stage: DatasetStageArgs (has .name and .data.dataset)
    - consumed_tokens_per_dataset_folder: dict[str,int] from previous stages
    - remaining_train_steps: steps planned for this stage
    """
    # Only main process prints
    if getattr(trainer, "parallel_context", None) is not None:
        try:
            is_main = trainer.is_main_process
        except Exception:
            is_main = True
    else:
        is_main = True
    if not is_main:
        return

    gbs = int(getattr(trainer, "global_batch_size", 1))
    seql = int(getattr(trainer, "sequence_length", 1))
    toks_per_step = gbs * seql
    plan_tokens = remaining_train_steps * toks_per_step
    cur_step = int(getattr(trainer, "iteration_step", 0))

    header = (
        f"\n[diagnostic] Stage start: {getattr(stage, 'name', 'Stage')}\n"
        f"  current_step={cur_step:,} | remaining_steps={remaining_train_steps:,}\n"
        f"  global_batch_size={gbs} samples/step | seq_len={seql} | tokens/step={toks_per_step:,}\n"
        f"  planned tokens this stage ≈ {plan_tokens:,}\n"
    )

    ds_args = stage.data.dataset
    if isinstance(ds_args, NanosetDatasetsArgs):
        folders = list(ds_args.dataset_folder)
        weights = list(getattr(ds_args, "dataset_weights", [1.0] * len(folders)))
        assert len(folders) == len(weights), "dataset_folder and dataset_weights must align"

        total_w = float(sum(weights)) if weights else 1.0
        if total_w == 0.0:
            weights = [1.0 / max(1, len(weights))] * len(weights)
            total_w = 1.0
        norm_w = [w / total_w for w in weights]

        wcol = max(12, max(len(p) for p in folders) + 2)
        header += (
            f"{'folder'.ljust(wcol)}  weight   start_offset_tokens  start_offset_samples"
            f"  expected_tokens  expected_samples  note"
        )
        log_rank(header, logger=logger, level=logging.INFO, rank=0)

        for path, w in zip(folders, norm_w):
            start_tok = int(consumed_tokens_per_dataset_folder.get(path, 0))
            start_smp = start_tok // seql
            exp_tok = int(round(plan_tokens * w))
            exp_smp = exp_tok // seql
            note = "NEW_in_stage" if start_tok == 0 else ""
            line = (
                f"{path.ljust(wcol)}  {w:6.4f}  {start_tok:>19,}  {start_smp:>19,}"
                f"  {exp_tok:>14,}  {exp_smp:>16,}  {note}"
            )
            log_rank(line, logger=logger, level=logging.INFO, rank=0)
        log_rank("", logger=logger, level=logging.INFO, rank=0)
    else:
        # Non-Nanoset datasets (HF pretrain or SFT) — print generic summary
        header += "  (non-Nanoset dataset; per-folder offsets not applicable)\n"
        log_rank(header, logger=logger, level=logging.INFO, rank=0)


# import lovely_tensors as lt
# lt.monkey_patch()


def _validate_rank_sidecar_metadata(rank_folders, *, expected_regime, seq_len, token_size, vocab_size):
    """Validate reference-rank sidecar metadata against the training config.

    The per-window checksum is computed over tokens only, so it cannot detect a sidecar scored
    in the wrong regime / seq_len / tokenizer — this catches those and fails loudly before training.
    """
    for folder in rank_folders:
        meta_files = sorted(glob.glob(f"{folder}/_rank_sidecar.*.metadata"))
        if not meta_files:
            raise ValueError(
                f"No _rank_sidecar.*.metadata found in reference-rank folder {folder}; (re)generate sidecars "
                f"with generate_rank_sidecar.py so the scoring regime can be validated."
            )
        for mf in meta_files:
            with open(mf) as f:
                meta = json.load(f)
            if meta.get("regime") != expected_regime:
                raise ValueError(
                    f"reference-rank sidecar regime mismatch in {mf}: scored '{meta.get('regime')}' but training "
                    f"expects '{expected_regime}'. Re-score with --regime {expected_regime}."
                )
            if meta.get("seq_len") != seq_len:
                raise ValueError(f"reference-rank sidecar seq_len mismatch in {mf}: {meta.get('seq_len')} != {seq_len}.")
            if meta.get("token_size") != token_size:
                raise ValueError(
                    f"reference-rank sidecar token_size mismatch in {mf}: {meta.get('token_size')} != {token_size}."
                )
            if vocab_size is not None and meta.get("vocab_size") is not None and meta["vocab_size"] < vocab_size:
                raise ValueError(
                    f"reference-rank teacher vocab_size {meta['vocab_size']} < trainee vocab_size {vocab_size} in "
                    f"{mf}: teacher cannot rank all trainee token ids (tokenizer mismatch?)."
                )


def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples: int,
    consumed_tokens_per_dataset_folder: Dict[str, int],
    num_remaining_train_steps: int,
    sanity_check_dataloader_interval: Optional[int] = None,
):
    """
    Returns a dataloader for a given data stage.

    data: The data configuration for the current stage.
    consumed_train_samples: The number of samples consumed by the model in the this stage (each stage starts from zero).
    num_remaining_train_steps: The number of remaining training steps for this stage.
    """
    assert consumed_train_samples >= 0, "consumed_train_samples should be greater than 0"
    assert num_remaining_train_steps >= 0, "num_remaining_train_steps should be greater than 0"

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Case 1: Dummy data generator
    if data.dataset is None:
        log_rank("Using dummy data generator", logger=logger, level=logging.INFO, rank=0)
        dataloader = dummy_infinite_data_generator(
            micro_batch_size=trainer.micro_batch_size,
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=trainer.model_config.vocab_size,
            seed=data.seed,
            parallel_context=trainer.parallel_context,
            use_position_ids=isinstance(
                trainer.model_config, Qwen2Config
            ),  # Simulate packed sequences to test SFT or inference
            cp_pg=trainer.parallel_context.cp_pg,
        )()

    # Case 2: HuggingFace datasets
    elif isinstance(data.dataset, PretrainDatasetsArgs) or isinstance(data.dataset, SFTDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # We need to the 1st device to process dataset and cache it, then other devices load from cache
        with main_rank_first(trainer.parallel_context.world_pg):
            # TODO @nouamanetazi: this may timeout before 1st device finishes processing dataset. Can we have a ctxmanager to modify timeout?
            # TODO: generalise to include  for validation/test splits

            # We load the raw dataset
            raw_dataset = get_datasets(
                hf_dataset_or_datasets=data.dataset.hf_dataset_or_datasets,
                hf_dataset_config_name=data.dataset.hf_dataset_config_name,
                splits=data.dataset.hf_dataset_splits,
            )["train"]

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.padding_side = "left"
            sequence_sep_tokens = [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, tokenizer.unk_token]
            # assert bos or eos are present
            assert tokenizer.bos_token is not None or tokenizer.eos_token is not None, (
                f"Tokenizer must have either bos or eos token, but found none for {tokenizer_path}"
            )

            # Check that tokenizer's vocab size is smaller than the model's vocab size
            assert tokenizer.vocab_size <= trainer.model_config.vocab_size, (
                f"Tokenizer's vocab size ({tokenizer.vocab_size}) is larger than the model's vocab size ({trainer.model_config.vocab_size})"
            )

            # Different processing for SFT vs pretraining
            if isinstance(data.dataset, SFTDatasetsArgs):
                # For SFT, use the dedicated prepare_sft_dataset function
                # Get optional debug parameter to limit dataset size (for faster development)
                debug_max_samples = getattr(data.dataset, "debug_max_samples", None)

                # Process the dataset using our dedicated SFT processing module
                train_dataset = prepare_sft_dataset(
                    raw_dataset=raw_dataset,
                    tokenizer=tokenizer,
                    trainer_sequence_length=trainer.sequence_length,
                    debug_max_samples=debug_max_samples,
                    num_proc=data.dataset.dataset_processing_num_proc_per_process,
                )
            else:
                # For pretraining, use existing CLM processing
                train_dataset = clm_process(
                    raw_dataset=raw_dataset,
                    tokenizer=tokenizer,
                    text_column_name=data.dataset.text_column_name,
                    dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
                    dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
                    sequence_length=trainer.sequence_length,
                )

            # We load the processed dataset on the ranks requiring it
            dataloader = get_train_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=consumed_train_samples,
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                dataloader_drop_last=True,
                use_position_ids=isinstance(trainer.model_config, Qwen2Config),
                sequence_sep_tokens=sequence_sep_tokens,  # Used to generate position ids
            )

            # Check if we have enough samples for train_steps
            total_tokens_dataset = len(dataloader.dataset) * trainer.sequence_length
            num_tokens_needed_for_training = (
                num_remaining_train_steps * trainer.global_batch_size * trainer.sequence_length
            )
            assert num_tokens_needed_for_training <= total_tokens_dataset, (
                f"Dataset is too small for steps ({total_tokens_dataset} < {num_tokens_needed_for_training}), "
                f"Try train_steps<={len(dataloader.dataset) // trainer.global_batch_size + trainer.iteration_step}"
            )

    # Case 3: Nanosets
    elif isinstance(data.dataset, NanosetDatasetsArgs):
        # Create Nanoset
        from nanotron.data.nanoset import Nanoset

        with main_rank_first(trainer.parallel_context.world_pg):
            tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
            # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            eos_token_id = 1
            assert eos_token_id is not None or data.dataset.return_positions is False, (
                "Tokenizer must have an eos token if return_positions is True"
            )
            log_rank(
                f"[Nanoset] Creating Nanoset with {len(data.dataset.dataset_folder)} dataset folders and {trainer.config.tokens.train_steps * trainer.global_batch_size} train samples",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            # Offline reference-model top-k masking: load the per-token teacher rank sidecars
            # (parallel to dataset_folder) iff the model is configured for that source.
            reference_offline = getattr(trainer.model_config, "topk_loss_mask_source", "self") == "reference_offline"
            # The Nanoset path runs cross-document attention; reference sidecars must be scored in
            # the matching regime (cross_doc <-> use_doc_masking=False). Tie both to one variable.
            use_doc_masking = False
            if reference_offline:
                if data.dataset.rank_dataset_folder is None:
                    raise ValueError(
                        "topk_loss_mask_source='reference_offline' requires data.dataset.rank_dataset_folder "
                        "(one rank sidecar folder per dataset_folder); otherwise training would run silently unmasked."
                    )
                _validate_rank_sidecar_metadata(
                    data.dataset.rank_dataset_folder,
                    expected_regime="doc_masked" if use_doc_masking else "cross_doc",
                    seq_len=trainer.sequence_length,
                    token_size=data.dataset.token_size_in_bytes,
                    vocab_size=data.dataset.vocab_size,
                )
            elif data.dataset.rank_dataset_folder is not None:
                log_rank(
                    "rank_dataset_folder is set but topk_loss_mask_source != 'reference_offline'; "
                    "the reference-rank sidecars will be ignored.",
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )
            start_time = time.time()
            train_dataset = Nanoset(
                dataset_folders=data.dataset.dataset_folder,
                sequence_length=trainer.sequence_length,
                token_size=data.dataset.token_size_in_bytes,
                train_split_num_samples=trainer.config.tokens.train_steps * trainer.global_batch_size,
                dataset_weights=data.dataset.dataset_weights,
                random_seed=data.seed,
                return_positions=data.dataset.return_positions,
                eos_token_id=eos_token_id,
                rank_dataset_folders=data.dataset.rank_dataset_folder if reference_offline else None,
            )
            end_time = time.time()
            log_rank(
                f"[Nanoset] Time taken to create Nanoset: {time.strftime('%M:%S', time.gmtime(end_time - start_time))} (MM:SS)",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
        # Prepare dataloader
        dataloader = build_nanoset_dataloader(
            train_dataset,
            trainer.sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            consumed_train_samples=consumed_train_samples,
            dataloader_num_workers=data.num_loading_workers,
            dataloader_drop_last=True,
            use_position_ids=True,
            use_doc_masking=use_doc_masking,
            emit_reference_ranks=reference_offline,
            dataloader_pin_memory=True,
        )
        dist.barrier()

    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}")

    if sanity_check_dataloader_interval is not None:
        sanity_check_dataloader(
            dataloader,
            tokenizer_path=trainer.config.tokenizer.tokenizer_name_or_path,
            sanity_check_dataloader_interval=sanity_check_dataloader_interval,
        )

    return dataloader


def get_dataloader(
    trainer: DistributedTrainer, sanity_check_dataloader_interval: Optional[int] = None
) -> Dict[str, DataLoader]:
    dataloaders = {}

    # Print training plan
    log_rank("Training plan", logger=logger, level=logging.INFO, rank=0, is_separator=True)
    stages_info = "".join(
        f"[Stage {stage.name}] start from step {stage.start_training_step} \n" for stage in trainer.config.data_stages
    )
    full_log_message = f"There are {len(trainer.config.data_stages)} training stages \n{stages_info}"
    log_rank(full_log_message, logger=logger, level=logging.INFO, rank=0)

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        (
            consumed_train_samples,
            consumed_tokens_per_dataset_folder,
        ) = get_consumed_train_samples_of_a_data_stage_from_ckp(stage, trainer.metadata)

        num_remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
            stage, trainer.config, trainer.metadata
        )
        log_rank(
            f"Stage {stage.name} has {num_remaining_train_steps} remaining training steps and has consumed {consumed_train_samples} samples"
            f"Consumed tokens per dataset folder: {pformat(consumed_tokens_per_dataset_folder)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        if stage_idx == 0:
            # Diagnostic print BEFORE building the stage-0 dataloader
            _log_stage_start_plan(
                trainer=trainer,
                stage=stage,
                consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
                remaining_train_steps=num_remaining_train_steps,
            )
            dataloader = get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
                num_remaining_train_steps=num_remaining_train_steps,
                sanity_check_dataloader_interval=sanity_check_dataloader_interval,
            )
        else:
            # Lazy init: wrap in a callable that logs diagnostics right before creating the loader
            def _make_stage_loader(
                stage=stage,
                consumed_train_samples=consumed_train_samples,
                consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
                num_remaining_train_steps=num_remaining_train_steps,
            ):
                _log_stage_start_plan(
                    trainer=trainer,
                    stage=stage,
                    consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
                    remaining_train_steps=num_remaining_train_steps,
                )
                return get_dataloader_from_data_stage(
                    trainer,
                    stage.data,
                    consumed_train_samples=consumed_train_samples,
                    consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
                    num_remaining_train_steps=num_remaining_train_steps,
                    sanity_check_dataloader_interval=sanity_check_dataloader_interval,
                )

            dataloader = _make_stage_loader

        dataloaders[stage.name] = dataloader
    return dataloaders


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    parser.add_argument(
        "--sanity-check-dataloader-interval",
        type=int,
        default=None,
        help="Optional interval to print dataloader samples",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    dataloader = get_dataloader(trainer, args.sanity_check_dataloader_interval)

    # Train
    trainer.train(dataloader)
