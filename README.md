# Online Dynamic Batching for HF Trainer

Train and evaluate a public multimodal fine-tuning example with
[Online Dynamic Batching](https://github.com/online-dynamic-batching/online-dynamic-batching)
and `transformers.Trainer`.

This repository is a runnable integration example. It is not a reproduction
package for the paper's experimental numbers; throughput and quality metrics
can differ with hardware, storage, dataset composition, model checkpoints, and
software versions.

## Prerequisites

- A Python environment with PyTorch and NVIDIA GPU support.
- A local or Hugging Face-accessible Qwen3-VL-2B-Instruct checkpoint, provided
  through `ODB_MM_MIX_MODEL` when you do not want to use the default model id.
- Network access to GitHub and the public data/model sources, or equivalent
  local mirrors.
- Enough disk space for the generated public TMDB data, checkpoints, validation
  outputs, and MMMU-MC benchmark outputs.

## Run ODB

Use a Python environment with PyTorch/GPU support, then run:

```bash
export ODB_MM_MIX_MODEL=/path/to/Qwen3-VL-2B-Instruct
./run.sh all-odb
```

This installs the example dependencies, builds the public data, trains the HF
Trainer ODB path, and runs validation loss plus MMMU-MC evaluation.

By default training uses one process. Set `ODB_MM_MIX_NUM_PROCESSES=8` for an
8-GPU run. The default training run uses a small public subset so the example
finishes quickly; set `ODB_MM_MIX_TRAIN_SIZE=0` to use the full public training
split.

## Tested Workflow

The tested workflow uses `online-dynamic-batching>=0.1.2`, Qwen3-VL-2B-Instruct,
the public MM-Mix TMDB recipe, and the LLaMA-Factory-compatible validation
split (`val_size=0.05`, `split_seed=42`). It covers:

- `./run.sh all-odb`: data build, ODB training, validation loss, and MMMU-MC.
- `./run.sh standard` plus `./run.sh eval-standard`: fixed-batch baseline
  training and evaluation.

The records under [results/](results/) are example run records.
They are useful for checking that the example behaves sensibly, but they should
not be read as paper-number reproduction results.

For stable benchmark runs, pre-cache MMMU locally and run evaluation with
`HF_DATASETS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.

## Run Step By Step

```bash
# Install ODB and the helper dependencies for this example.
./run.sh install

# Download/build the public multimodal TMDB training data.
./run.sh data

# Inspect the lazy HF processor path before training.
./run.sh inspect

# Train the recommended ODB path and save the final checkpoint for evaluation.
ODB_MM_MIX_SAVE_FINAL_MODEL=1 ./run.sh odb-enable

# Compute validation loss and MMMU-MC for the ODB checkpoint.
./run.sh eval-odb
```

The default paths are:

- Public data: `data/mm-mix-tmdb`
- Shared data recipe checkout: `.deps/odb-mm-mix-example`
- Checkpoints and eval outputs: `outputs/hf-trainer-real`

## Run Standard

After `./run.sh install` and `./run.sh data`, run the fixed-batch baseline:

```bash
ODB_MM_MIX_SAVE_FINAL_MODEL=1 ./run.sh standard
./run.sh eval-standard
```

## Common Options

```bash
# Use 8 GPUs through torch.distributed.run.
ODB_MM_MIX_NUM_PROCESSES=8 ./run.sh odb-enable

# Pick a different launcher port when running multiple jobs on one machine.
ODB_MM_MIX_MASTER_PORT=29681 ./run.sh odb-enable

# Use the full public training split.
ODB_MM_MIX_TRAIN_SIZE=0 ./run.sh odb-enable

# Save a checkpoint for validation loss and benchmark evaluation.
ODB_MM_MIX_SAVE_FINAL_MODEL=1 ./run.sh odb-enable

# Tune the image cap for larger or smaller visual inputs.
ODB_MM_MIX_IMAGE_MAX_PIXELS=589824 ./run.sh odb-enable

# Evaluate a custom checkpoint.
ODB_HF_EVAL_CHECKPOINT=/path/to/checkpoint ./run.sh eval-valloss
ODB_HF_EVAL_CHECKPOINT=/path/to/checkpoint ./run.sh benchmark
```

## Outputs

Default model directories:

| Target | Directory |
| --- | --- |
| ODB one-call hook | `outputs/hf-trainer-real/odb-enable` |
| ODB manual bridge | `outputs/hf-trainer-real/odb-manual` |
| Standard | `outputs/hf-trainer-real/standard-none` |

Validation-loss outputs are written under the evaluated checkpoint directory as
`eval_out_hf_valloss`.

MMMU-MC outputs are written under the evaluated checkpoint directory as
`mmmu_mc_likelihood_hf` and include:

- `mmmu_mc_likelihood_results.json`
- `predictions.jsonl`
- `excluded.jsonl`
- `score_audit.json`

## Commands

| Command | Purpose |
| --- | --- |
| `./run.sh install` | Install Python dependencies for this example. |
| `./run.sh data` | Build the public TMDB data. |
| `./run.sh inspect` | Inspect HF processor multimodal tensor output. |
| `./run.sh odb-enable` | Train with the recommended `enable_odb(...)` path. |
| `./run.sh odb-manual` | Train with explicit `odb.apply(...)` + `configure_trainer(...)`. |
| `./run.sh standard` | Train the fixed-batch baseline. |
| `./run.sh eval-odb` | Evaluate the ODB checkpoint. |
| `./run.sh eval-standard` | Evaluate the Standard checkpoint. |
| `./run.sh eval-valloss` | Evaluate validation loss for a saved checkpoint. |
| `./run.sh benchmark` | Run the built-in MMMU-MC benchmark. |
| `./run.sh all-odb` | Run the complete ODB path. |

## Integration Notes

This example supports two ODB integration styles:

| Mode | Command | What it demonstrates |
| --- | --- | --- |
| One-call hook | `./run.sh odb-enable` | Recommended `enable_odb(...)` entrypoint for ODB-ready HF Trainer pipelines. |
| Manual bridge | `./run.sh odb-manual` | Lower-level `odb.apply(...)` plus `configure_trainer(...)`, useful when you want explicit control over the DataLoader handle. |

HF Trainer can train multimodal models once each batch contains the tensors
expected by `model.forward`. ODB adds one extra contract: model-specific
tokenization, image processing, and visual-token expansion must happen before
ODB grouping, so ODB can see the true post-processing length of each sample.
In this example, the lazy Dataset returns single-sample tensor dictionaries and
the collator only pads/stacks tensors.

The default image cap is chosen for stable out-of-the-box execution. Tune
`ODB_MM_MIX_IMAGE_MAX_PIXELS` when you want to allow larger images.

## Related Examples

- Shared public data recipe: [odb-mm-mix-example](https://github.com/online-dynamic-batching/odb-mm-mix-example)
- LLaMA-Factory example: [odb-example-llamafactory](https://github.com/online-dynamic-batching/odb-example-llamafactory)
- Accelerate example: [odb-example-accelerate](https://github.com/online-dynamic-batching/odb-example-accelerate)
- Lightning example: [odb-example-lightning](https://github.com/online-dynamic-batching/odb-example-lightning)
