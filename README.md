# A Training-Free Pipeline for the AI Coach Cooking Challenge

Training-free, single-GPU pipeline for the **AI Coach Challenge: Cooking** at
the VAR Workshop, CVPR 2026. Built upon a Qwen3-VL-8B-Instruct base, the
system turns mistake detection into a multiple-choice problem grounded by an
agentic Grounding DINO visual verifier.

## Abstract

We present a training-free pipeline for the AI Coach Cooking Challenge,
built upon the Qwen3-VL-8B-Instruct baseline. In its default free-form
setting, this baseline suffers from a strong *"no mistake"* bias, completely
failing to detect errors (F1 = 0.00). Our framework overcomes this
limitation to address fine-grained cooking recipes and mistakes without
requiring any model fine-tuning. First, following Bhattacharyya et al.
(NeurIPS 2025), we generate a checklist of plausible mistakes for each
instruction. We then provide the model with the video frames and this
checklist, formulating the error-detection task as a multiple-choice
selection. Subsequently, we employ an agentic visual verifier (Grounding
DINO) via tool-calling to post-check whether the object nouns referenced
in the selected mistakes are actually visible in the frame; predictions
lacking visual support are dropped. This verification step significantly
reduces false positives. On the official `main` / `test` split, our system
achieves an F1 score of **0.20** and an IC-Acc of **31.4**. This represents
a substantial improvement over the 8B baseline (F1 = 0.00, IC-Acc = 19.8),
all accomplished with zero training on a single GPU.

## Results

Reported on the official `plan_set=main`, `split=test` configuration of the
Qualcomm Interactive Cooking benchmark. 

| System | IC-Acc ↑ | Prec. ↑ | Rec. ↑ | F1 ↑ | BERT ↑ | ROUGE-L ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | 19.78 | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 |
| **Ours (MCQ + Grounding DINO)** | **31.45** | 0.17 | 0.25 | **0.20** | **0.450** | **0.336** |

All metrics target the **mistake stream** (Feedback emits matched against
ground-truth mistakes), except **IC-Acc** which scores instruction-completion
accuracy. F1 is computed from the segment-level (TP, FP, TN, FN) counts that
the official evaluator produces; BERT and ROUGE-L are averaged over the text
of true-positive Feedback emits only.

## Pipeline

1. **Per-instruction mistake checklist** — for each ground-truth instruction,
   we precompute a short list of plausible mistakes following the
   self-consistency / summary scheme of Bhattacharyya et al. (NeurIPS 2025).
   See `assets/mistake_summaries_main_test.json`.
2. **Multiple-choice mistake detection** — at each sliding window, the VLM
   is shown the checklist and asked which numbered item is happening
   (or `N` for none). This breaks the model's free-form *"no mistake"*
   bias and forces a constrained decision.
3. **Agentic Grounding DINO verifier** — if the model picks a mistake,
   we look up the object nouns referenced in that mistake
   (`assets/mistake_objects_main_test.json`), then call Grounding DINO on
   the current frame. If the named objects aren't visually present, the
   prediction is dropped.

## Setup

```bash
conda create --name aicoach python=3.11.10
conda activate aicoach
conda install bert_score rouge-score tqdm -c conda-forge
pip3 install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```


## Data

The pipeline operates on the [CaptainCook4D](https://github.com/CaptainCook4D/captaincook4d)
GoPro videos. Annotations come from the
[Qualcomm Interactive Cooking benchmark](https://huggingface.co/datasets/qualcomm/Qualcomm-Interactive-Cooking)
on HuggingFace and are loaded automatically by `data.py`.

Set `<DATA_ROOT>` to the directory containing `<video_id>_360p.mp4` files,
then extract frames at 2 FPS:

```bash
python extract_frames.py --captaincook4d_root <DATA_ROOT> --fps 2
```

This writes one JPEG per sampled frame to
`<DATA_ROOT>/resolution_360p_video_frames_2fps/<video_id>_360p/frame_XXXXXX.jpg`.

## Reproducing the result

The repository ships the precomputed mistake summaries and object lists
used for our final submission, so you can reproduce the F1 = 0.20 number
without re-running the offline summary stage.

```bash
export HF_HOME=<path to hf cache>

python inference.py \
    --captaincook4d_root <DATA_ROOT> \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --plan_set main --split test \
    --turn_based \
    --prompt_config_file prompts/paper_full.json \
    --mistake_summary_file assets/mistake_summaries_main_test.json \
    --mcq_mistake_inference \
    --max_feedback_per_segment 2 \
    --dedupe_feedback_text \
    --grounding_dino_verify \
    --mistake_objects_file assets/mistake_objects_main_test.json \
    --predictions_save_root ./predictions \
    --predictions_file_name predictions.json
```

Flag summary:

| Flag | Purpose |
|---|---|
| `--prompt_config_file prompts/paper_full.json` | Paper-style `instruction_end` / `mistake_inference` prompts |
| `--mistake_summary_file ...` | Per-instruction mistake checklist (Bhattacharyya et al.) |
| `--mcq_mistake_inference` | Multiple-choice mistake selection from the checklist |
| `--max_feedback_per_segment 2` | Cap intra-segment Feedback repeats |
| `--dedupe_feedback_text` | Drop duplicate Feedback within a segment |
| `--grounding_dino_verify` | Agentic Grounding DINO object-presence check |
| `--mistake_objects_file ...` | Per-mistake object noun lookup table for DINO |

## Evaluation

Use the [official evaluation repo](https://github.com/Qualcomm-AI-research/qualcomm_interactive_cooking_eval):

```bash
PYTHONPATH=./qualcomm_interactive_cooking_eval python qualcomm_interactive_cooking_eval/eval.py \
    --plan_set main \
    --split test \
    --predictions_file_path ./predictions/predictions.json
```

## Rebuilding the assets

If you'd like to regenerate the per-instruction summaries and object lists
from scratch (e.g., for a different split), the offline tools live in
`tools/`:

```bash
# 1) Per-instruction mistake summary (uses an LLM to expand each instruction
#    into a checklist of plausible mistakes; resumable).
python tools/build_mistake_summaries.py \
    --plan_set main --split test \
    --output_path assets/mistake_summaries_main_test.json

# 2) Object-noun lookup table for the DINO verifier (extracts the salient
#    nouns from each summary item).
python tools/extract_summary_objects.py \
    --summaries_file assets/mistake_summaries_main_test.json \
    --output_path assets/mistake_objects_main_test.json
```

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data.py                 # Qualcomm benchmark + CaptainCook4D loader
├── extract_frames.py       # 2-FPS frame extractor
├── utils.py
├── inference.py            # Main inference entry point
├── prompts/
│   └── paper_full.json     # Paper-style prompts
├── assets/
│   ├── mistake_summaries_main_test.json
│   └── mistake_objects_main_test.json
└── tools/
    ├── build_mistake_summaries.py
    ├── extract_summary_objects.py
    └── grounding_dino_verifier.py
```

## Acknowledgements

The pipeline builds on:

- **VAR Workshop @ CVPR 2026 — AI Coach Cooking baseline**
  ([varworkshop/ai_coach_cooking_2026](https://github.com/varworkshop/ai_coach_cooking_2026)).
  The starter code that this submission was forked from: data loaders,
  frame extraction, the original Qwen3-VL inference loop, and the
  `instruction_end` / `mistake_inference` prompt skeleton.
- **Bhattacharyya et al., NeurIPS 2025** — per-instruction mistake summary
  formulation.
- **Qwen3-VL-8B-Instruct** — base vision-language model
  ([Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)).
- **Grounding DINO** — open-set object detector used as the agentic
  visual verifier ([paper](https://arxiv.org/abs/2303.05499)).
- **Qualcomm Interactive Cooking benchmark** — task definition, splits,
  and evaluation.
