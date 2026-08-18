# Dataset contract for Codex

`train.py` intentionally leaves MovieNet pair parsing to a local file that Codex can implement after inspecting the generated pair data:

```text
experiments/stage_2/ariadnet_lut_neural_preset_setting_movienet_pair_stage_2/
  movienet_pair_dataset.py
```

Required API:

```python
build_loader(cfg, split: str, train: bool) -> tuple[DataLoader, Dataset]
```

Required batch keys (float tensors in `[0,1]`, shape `[B,3,256,256]`):

```python
{
    "frame_a":      original movie frame A,
    "frame_b":      original movie frame B,
    "corrupted_a":  independently color-corrupted frame A,
    "corrupted_b":  independently color-corrupted frame B,
}
```

Optional metadata keys are welcome (`movie_id`, paths, pair id, Qwen/proxy scores, corruption LUT paths), but training does not require them.

## Exact training relation

For every accepted same-style / different-content movie pair `(A,B)`:

```text
corrupted(A) + reference B -> GT A
corrupted(B) + reference A -> GT B
```

`train.py` concatenates these two directions into one forward pass.

## Very important corruption rule

Use the **same 308 `.cube` LUT distribution as Stage-1 Neural Preset training** (`cfg.data.lut_root`). The Stage-1 normalizer is frozen, therefore feeding a completely new corruption family would unnecessarily create another train/test mismatch.

For training:

- sample the corruption LUT for A and B independently;
- resample on every dataset access / epoch;
- do not use the reference frame's color as the corruption target;
- keep original movie frame exactly as pixel GT.

For validation:

- deterministic LUT choice based on `cfg.data.validation_seed + pair_index`;
- identical result across runs.

## Recoverability filter

A very aggressive LUT can clip enough pixels that reconstructing the original movie frame is information-theoretically impossible. After applying a candidate corruption LUT, compute a simple clipping fraction (e.g. pixels/channels very near 0 or 1 compared with the uncorrupted frame). If it exceeds:

```yaml
max_corruption_clipped_fraction: 0.02
```

resample the LUT, up to `max_corruption_resample_attempts`.

The filter should be conservative: the goal is not to make corruption easy, only to reject obviously non-recoverable cases.

## Preprocessing

For the first controlled experiment match Stage-1 as closely as possible:

- RGB
- direct bicubic resize to `256x256`
- no crop
- no horizontal flip
- tensor range `[0,1]`

Do not add geometric augmentation in this first run because GT is exact and the pair-selection experiment should remain easy to interpret.
