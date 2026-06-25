---
name: bracket-dataset-toml
description: Use when the user wants to create a dataset.toml for a Bracket / musubi-tuner / sd-scripts training session, or asks "how do I configure my dataset", "what does the toml file look like", "I have a folder of images, what's next". Walks through asking only the questions the toml format actually needs (paths, resolution, repeats), validates the result, and writes the file.
---

# Bracket — building a dataset.toml interactively

A musubi-tuner / sd-scripts dataset.toml describes one or more
"datasets" (or "subsets" inside a dataset). Each subset points at a
directory of images plus matching `.txt` caption files (or a single
metadata JSONL). The trainer reads this to know what to train on,
what resolution to pack into, and how many repeats per epoch.

## When to invoke this skill

Use this skill when:
- The user has a folder of images and asks how to feed them to
  Bracket.
- The user says "I don't have a dataset.toml" / "how do I write one".
- A `bracket-run-session` flow needs a dataset.toml and the user
  hasn't supplied one.

## What you need from the user (ask in order, ONE at a time)

1. **Image directory.** Absolute path to the folder containing the
   training images.
2. **Caption format.** Two options:
   - **`.txt` per image** (most common). Same basename as the image,
     e.g. `0001.jpg` + `0001.txt`. Caption is the prompt, one line.
   - **Single metadata JSONL/JSON.** Path to the file. Each row
     contains the image path + caption.
3. **Resolution.** Trainer will bucket-pack — ask which side they
   want as the target, then pass to the trainer:
   - SDXL / SD3.5 → 1024
   - Flux.1 / Flux.1-Kontext → 1024 (Kontext sometimes 768)
   - Flux-2-Klein / Z-Image / Qwen-Image → 1024 or 1280
   - Wan / HunyuanVideo / LTX (video) → 480 / 544 / 720 (depends on
     VRAM and target output resolution); also need `--frame_extraction`
     and `target_frames`.
4. **Number of repeats.** How many times to iterate over each image
   per epoch. 1 for big datasets (>500 images), 4-10 for tiny
   (10-50). Default to 1 if unsure.
5. **Batch size.** 1 is safe everywhere. 2-4 only if VRAM allows
   (mostly SDXL LoRA). For all musubi flow-matching trainers (Z-Image,
   Flux, Qwen-Image, Wan, Hunyuan) **stay at 1** — Bracket's search
   space pins this and varies gradient_accumulation_steps instead.
6. **Where to write the file.** Default to
   `./configs/<dataset-name>.toml`.

## Image-only template (most users)

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/abs/path/to/images"
num_repeats = 1
```

That's a working dataset.toml for SDXL / Z-Image / Flux.1 /
Flux-2-Klein / Qwen-Image / SD3.5 LoRA.

## Image-edit template (Flux.1-Kontext, Qwen-Image-Edit)

These trainers need PAIRED source + target directories (the model
learns to edit `source` into `target`):

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
[[datasets.source]]
image_directory = "/abs/path/to/source-images"
num_repeats = 1

[[datasets.target]]
image_directory = "/abs/path/to/target-images"
num_repeats = 1
```

Source and target images must have matching filenames.

## Video template (Wan, HunyuanVideo, LTX-2, FramePack)

```toml
[general]
resolution = [544, 960]   # H, W — keep 16:9 or 9:16; multiples of 16
caption_extension = ".txt"
batch_size = 1
enable_bucket = false     # bucketing is per-resolution-set in video

[[datasets]]
video_directory = "/abs/path/to/videos"
num_repeats = 1
target_frames = [1, 25, 49]   # frame counts to bucket into
frame_extraction = "head"     # "head" | "chunk" | "uniform"
source_fps = 30
```

For FramePack add `latent_anchor_frames = [1]` to the dataset.

## Metadata JSONL template (alternative to .txt-per-image)

```toml
[general]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true

[[datasets]]
metadata_file = "/abs/path/to/metadata.jsonl"
num_repeats = 1
```

JSONL row format: `{"file_name": "0001.jpg", "caption": "..."}`.

## Multiple subsets

Each `[[datasets]]` can be repeated. Useful when mixing styles or
weighting:

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
image_directory = "/abs/path/to/portraits"
num_repeats = 4    # heavier weight

[[datasets]]
image_directory = "/abs/path/to/landscapes"
num_repeats = 1
```

## Validate before writing

Before you `Write` the file:

1. Check the `image_directory` exists (use `Bash: ls`).
2. Spot-check 3-5 images have matching `.txt` files (if
   `caption_extension = ".txt"`).
3. Reject batch_size > 1 with a warning for any musubi flow-matching
   trainer family.
4. Reject `enable_bucket = true` with `resolution = [H, W]` where H ≠
   W and they're not using video — square bucketing is sane;
   non-square needs `enable_bucket = false` with a fixed crop, OR
   uses musubi's separate bucket_resolution_steps (advanced, ask the
   user explicitly).

## After writing

1. Print the absolute path of the written file.
2. Tell the user they can pass it to a session via the Setup tab's
   "Dataset config TOML" field, or via CLI as
   `--dataset-toml <path>`.
3. Offer to run a quick verification: load the file via
   `python -c "import toml; print(toml.load('<path>'))"` to catch
   syntax errors.

## Anti-patterns

- Don't write a dataset.toml without asking for the image directory
  first — it's the one piece you can't infer.
- Don't auto-pick num_repeats > 1 for a dataset with > 200 images
  unless the user explicitly asks. It just wastes wall time.
- Don't include knobs the user hasn't been asked about
  (`color_aug`, `flip_aug`, `face_crop_aug_range`, etc.) — most
  augmentations actively hurt fine-tune quality and Bracket's search
  space doesn't expose them.
- Don't write to a path under `vendor/` or `runs/`. Default to
  `configs/` or wherever the user explicitly says.
