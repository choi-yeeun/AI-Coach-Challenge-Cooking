from typing import Any, List, Dict, Optional
import numpy.typing as npt
import itertools
import random
import json
import torch
import numpy as np
from PIL import Image
import os
import sys
import logging
import argparse
import multiprocessing as mp
from dataclasses import dataclass, field

from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, Qwen3VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from tqdm import tqdm

from data import QualcommInteractiveCookingDatasetVideos
from utils import load_frames_into_array


@dataclass
class PromptConfig:
    """Holds a system prompt and a user text template for a given inference mode."""

    system_prompt: str
    user_text_template: str

    def format_user_text(self, **kwargs) -> str:
        """Format the user text template with the provided keyword arguments."""
        return self.user_text_template.format(**kwargs)


PROMPT_CONFIGS: Dict[str, PromptConfig] = {
    "instruction_end": PromptConfig(
        system_prompt=(
            "You are an expert cooking assistant who is observing a person cook. "
        ),
        user_text_template=(
            "The person is currently at the following recipe step: {instruction} "
            "Has the person already completed the recipe step?  "
            "If the person has completed the recipe step answer 'YES' else answer 'NO'."
            "If you answer 'YES' describe why you think the person already completed the recipe step. "
        ),
    ),
    "mistake_inference": PromptConfig(
        system_prompt=(
            "You are an expert cooking assistant who is observing a person cook. "
            "You should look out for mistakes made by the person. "
        ),
        user_text_template=(
            "The person is trying to complete the following recipe step: {instruction}  "
            "Your task is to check if the person is about to make or has already made a mistake. "
            "Mistakes occur when the person performs actions that deviates from the instruction and DIRECTLY INTERFERES WITH SUCCESSFUL INSTRUCTION COMPLETION. "
            "Do not penalize actions that do not directly interfere with instruction completion (e.g. washing broccoli before cutting, if the instruction just says 'cut broccoli'). "
            "Here are some common types of mistakes that you should look out for: \n\n"
            "1. Technique Error: A mistake in how a step is physically performed. "
            "Examples include chopping with the wrong motion, stirring when folding is required, or spilling during transfer, "
            "producing uneven cuts or texture issues even when tools and amounts are right.\n"
            "Ignore minor technique errors such as not holding or gripping objects properly, holding with a risk of dropping etc. "
            "that do not interfere directly with recipe completion.\n "
            "2. Preparation Error: A setup mistake before executing the step . "
            "Using the wrong or dirty utensil, not washing/peeling/draining ingredients, insufficient draining of fluid, cutting/ chopping without peeling which makes correct execution difficult or unsafe.\n"
            "3. Measurement Error: An error in quantity — wrong counts, volumes, weights, or units. Mixing up teaspoons and tablespoons, "
            "misreading a scale, or miscounting items leads to off ratios and predictable taste or texture problems."
            "4. Temperature Error: A mistake in heat level or thermal state — the applied temperature, starting temperature, or thermal transition is wrong. "
            "Not preheating, using the wrong microwave power, overheating oil, or adding cold liquid when warm is required often causes burning, undercooking, or split emulsions."
            "5. Timing Error: A mistake in duration -- over- or under-doing a step or skipping required rests, proofs, or cooling periods. "
            "Overcooking, underblending, or cutting resting time short typically yields incorrect doneness or unstable textures.\n\n"
            "Assume the recipe step is still in progress. Your task is to identify any mistake that's already visible in the partially completed step. "
            "Do no penalize partially competed recipe steps. "
            "If you observe a mistake answer 'YES', else 'No'. "
            "Your response MUST BEGIN WITH 'YES' or 'NO'. In case you answer 'YES', please follow with a concise feedback to the user describing the mistake (i.e. YES. <feedback>.). Directly address the person."
        ),
    ),
    # Multiple-choice mistake detection. The numbered ``{mcq_options}`` are
    # derived from the per-instruction mistake summary (one option per listed
    # mistake). The model is forced to pick a digit or 'N' for no mistake —
    # no free-form generation, no hallucinated text. Selected text is then
    # looked up verbatim from the summary and emitted as the Feedback string.
    "mistake_inference_mcq": PromptConfig(
        system_prompt=(
            "You are an expert cooking assistant helping a person cook. "
            "The person is provided with an instruction and your task is to "
            "identify a specific mistake from a fixed checklist."
        ),
        user_text_template=(
            "## INSTRUCTIONS:\n"
            "The person has been instructed to: {instruction}.\n"
            "You are checking ONLY for these specific mistakes:\n"
            "{mcq_options}\n\n"
            "Which mistake is the person making? Answer with the NUMBER (digit) "
            "of the matching mistake, or 'N' if no mistake from the list is "
            "happening.\n"
            "ONE CHARACTER ONLY."
        ),
    ),
    # Variant used when an instruction-specific mistake summary is available
    # (produced offline by tools/build_mistake_summaries.py). The structure
    # mirrors ``mistake_inference`` — the generic 5-category error list is
    # swapped for a per-instruction summary injected via ``{mistake_summary}``.
    "mistake_inference_with_summary": PromptConfig(
        system_prompt=(
            "You are an expert cooking assistant who is observing a person cook. "
            "You should look out for mistakes made by the person. "
        ),
        user_text_template=(
            "The person is trying to complete the following recipe step: {instruction}  "
            "Your task is to check if the person is about to make or has already made a mistake. "
            "Mistakes occur when the person performs actions that deviates from the instruction and DIRECTLY INTERFERES WITH SUCCESSFUL INSTRUCTION COMPLETION. "
            "Do not penalize actions that do not directly interfere with instruction completion (e.g. washing broccoli before cutting, if the instruction just says 'cut broccoli'). "
            "This is how you can check for mistakes: \n\n"
            "{mistake_summary}\n\n"
            "Assume the recipe step is still in progress. Your task is to identify any mistake that's already visible in the partially completed step. "
            "Do no penalize partially competed recipe steps. "
            "If you observe a mistake answer 'YES', else 'No'. "
            "Your response MUST BEGIN WITH 'YES' or 'NO'. In case you answer 'YES', please follow with a concise feedback to the user describing the mistake (i.e. YES. <feedback>.). Directly address the person."
        ),
    ),
}
    

def build_messages(mode: str, **kwargs) -> List[Dict]:
    """Build the chat message list for the given prompt mode and template variables."""
    if mode not in PROMPT_CONFIGS:
        raise ValueError(f"Unknown prompt mode: {mode}. Available modes: {list(PROMPT_CONFIGS.keys())}")

    config = PROMPT_CONFIGS[mode]
    user_text = config.format_user_text(**kwargs)

    messages = [
        {
            "role": "system",
            "content": config.system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                },
                {
                    "type": "text",
                    "text": user_text,
                },
            ],
        },
    ]
    return messages


def load_prompt_configs_from_file(path: str) -> None:
    """Load and register prompt configs from a JSON file, overriding any existing entries."""
    with open(path, "r") as f:
        raw = json.load(f)
    for mode, cfg in raw.items():
        PROMPT_CONFIGS[mode] = PromptConfig(
            system_prompt=cfg["system_prompt"],
            user_text_template=cfg["user_text_template"],
    )


# Module-level cache of {instruction_text: mistake_summary_text}.
# Populated by ``load_mistake_summaries`` when ``--mistake_summary_file`` is set.
# Each worker process loads its own copy (spawn).
_MISTAKE_SUMMARIES: Dict[str, str] = {}


def parse_summary_items(summary_text: str) -> List[tuple]:
    """Parse a numbered summary string into [(n:int, mistake_text:str), ...].

    Expects lines of the form '1. ...', '2. ...' (as produced by
    tools/build_mistake_summaries.py). Lines that don't match are skipped.
    """
    import re
    items: List[tuple] = []
    for line in summary_text.splitlines():
        m = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", line)
        if m:
            items.append((int(m.group(1)), m.group(2).strip()))
    return items


def parse_mcq_response(text: str, items: List[tuple]) -> Optional[str]:
    """Return the chosen mistake text from ``items`` (verbatim), or None.

    Looks at the first non-whitespace character of ``text``. 'N' / 'n' (or
    any non-digit) → None (no mistake). A digit → lookup in ``items`` by
    matching number; returns the corresponding text, or None if out of range.
    """
    text = text.strip()
    if not text:
        return None
    first = text[0]
    if first.isdigit():
        d = int(first)
        for num, mistake in items:
            if num == d:
                return mistake
        return None  # digit out of listed range
    return None  # 'N' or anything else


def load_mistake_summaries(path: str) -> None:
    """Load instruction-specific mistake summaries from JSON and cache them globally.

    The JSON is expected to be a flat ``{instruction_text: summary_text}`` map,
    typically produced by ``tools/build_mistake_summaries.py``.
    """
    global _MISTAKE_SUMMARIES
    with open(path, "r") as f:
        _MISTAKE_SUMMARIES = json.load(f)
    if not isinstance(_MISTAKE_SUMMARIES, dict):
        raise ValueError(
            f"mistake_summary_file must be a dict, got {type(_MISTAKE_SUMMARIES).__name__}"
        )
    print(f"[mistake_summary] loaded {len(_MISTAKE_SUMMARIES)} entries from {path}")


# Module-level cache of {summary_item_text: [object_noun, ...]} for
# --grounding_dino_verify. Produced offline by tools/extract_summary_objects.py.
# Lookup is by VERBATIM summary item text (MCQ emits items verbatim) with a
# whitespace-normalized fallback. Empty list = no concrete object → keep emit.
_MISTAKE_OBJECTS: Dict[str, List[str]] = {}
_MISTAKE_OBJECTS_NORMALIZED: Dict[str, List[str]] = {}


def load_mistake_objects(path: str) -> None:
    """Load {summary_item: [object_noun, ...]} from JSON and build the lookup dict."""
    global _MISTAKE_OBJECTS, _MISTAKE_OBJECTS_NORMALIZED
    with open(path, "r") as f:
        _MISTAKE_OBJECTS = json.load(f)
    if not isinstance(_MISTAKE_OBJECTS, dict):
        raise ValueError(
            f"mistake_objects_file must be a dict, got {type(_MISTAKE_OBJECTS).__name__}"
        )
    _MISTAKE_OBJECTS_NORMALIZED = {
        " ".join(k.split()).lower(): v for k, v in _MISTAKE_OBJECTS.items()
    }
    print(f"[mistake_objects] loaded {len(_MISTAKE_OBJECTS)} entries from {path}")


def lookup_objects(feedback_text: str) -> List[str]:
    """Lookup expected objects for a Feedback string. Tries verbatim then normalized."""
    if feedback_text in _MISTAKE_OBJECTS:
        return _MISTAKE_OBJECTS[feedback_text]
    norm = " ".join(feedback_text.split()).lower()
    return _MISTAKE_OBJECTS_NORMALIZED.get(norm, [])


def load_model_and_processor(model_id, cache_dir=None, device_map="cuda:0"):
    """Load the Qwen3-VL model and its processor from HuggingFace."""
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=device_map
    ).eval()
    print("Model device map:", model.hf_device_map)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained( model_id,
        cache_dir=cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=device_map
    ).eval()
    print("Model device map:", model.hf_device_map)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor
    


@torch.no_grad()
def get_qwen_vl_output(
    mode: str,
    model,
    processor,
    video=None,
    instruction: str = None,
    mistake_summary: Optional[str] = None,
    mcq_options: Optional[str] = None,
    max_new_tokens: int = 128,
) -> str:
    """
    Run a single inference pass with Qwen3-VL.

    Args:
        mode: Prompt mode key (must exist in PROMPT_CONFIGS).
        model: The loaded Qwen3-VL model.
        processor: The associated processor.
        video: Sequence of video frames to pass as input.
        instruction: Recipe instruction text to embed in the prompt.
        mistake_summary: Optional per-instruction mistake summary, injected into
            templates that have a ``{mistake_summary}`` slot
            (e.g. ``mistake_inference_with_summary``).
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Decoded output string from the model.
    """
    assert mode in PROMPT_CONFIGS, f"Unknown mode: {mode}"

    template_kwargs: Dict[str, Any] = {"instruction": instruction}
    if mistake_summary is not None:
        template_kwargs["mistake_summary"] = mistake_summary
    if mcq_options is not None:
        template_kwargs["mcq_options"] = mcq_options
    messages = build_messages(mode, **template_kwargs)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    video_metadata = {"fps": 2, "total_num_frames": len(video)}
    inputs = processor(
        text=[text],
        images=None,
        videos=video,
        padding=True,
        video_metadata=[video_metadata],
        return_tensors="pt",
    )
    inputs = inputs.to("cuda").to(torch.bfloat16)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]


def _setup_logger(log_file_path: str, also_stdout: bool = True, mode: str = "w") -> logging.Logger:
    """Configure a root logger that writes to file (and optionally stdout). Idempotent.

    ``mode`` is the open mode for the FileHandler — pass ``"a"`` when resuming
    so prior run's log content is preserved.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handlers: List[logging.Handler] = [logging.FileHandler(log_file_path, mode=mode)]
    if also_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=handlers,
    )
    return logging.getLogger()


def process_one_video(
    eval_idx, data, video, model, processor, args, logger, inner_pbar: bool = True,
    verifier=None,
) -> Dict[str, Any]:
    """Run the baseline inference loop on a single video and return its predictions dict.

    Extracted from the original ``run`` loop so that both the single-process path
    and the multi-process workers share the exact same logic.

    ``inner_pbar`` controls whether the per-video tqdm bar is drawn. Multi-process
    workers should pass ``inner_pbar=False`` so their bars don't clobber each
    other in the shared terminal.
    """
    curr_video_predictions_to_save = {
        "video_id": data["video_id"],
        "pred_texts": [],
        "pred_timestamps": [],
    }

    # initialize video buffer
    action_idx = 0
    init_video_seek = int(
        (data["global_start_timestamp"] - data["video_frame_timestamps"][0]) * args.video_fps
    )
    video_buffer_start_index = init_video_seek
    video_buffer_seek_index = init_video_seek + args.video_seek_amount

    pbar = tqdm(
        total=len(data["video_frame_paths"]) - init_video_seek,
        desc=f"video {data['video_id']}",
        leave=False,
        disable=not inner_pbar,
    )

    while action_idx < data["num_of_instruction_segments"] and \
            video_buffer_seek_index < len(data['video_frame_paths']):
        logger.info("-*" * 40)
        gt_instruction = data["gt_texts"][action_idx][0]
        gt_is_mistake = data["gt_has_mistake"][action_idx]
        gt_mistakes_compiled = [x for x in data["gt_texts"][action_idx] if "Feedback" in x]

        gt_instruction_start_time = data["gt_text_timestamps"][action_idx][0]
        gt_instruction_end_time = data["gt_text_timestamps"][action_idx][-1]

        # Per-segment Feedback counter for --max_feedback_per_segment.
        # Reset every time we enter a new instruction segment.
        feedbacks_emitted_this_segment = 0
        # Per-segment set of normalized Feedback texts for --dedupe_feedback_text.
        emitted_feedback_texts_this_segment: set = set()

        if args.turn_based:
            curr_video_predictions_to_save["pred_texts"].append("Instruction: " + gt_instruction)
            curr_video_predictions_to_save["pred_timestamps"].append(gt_instruction_start_time)

            # restart video stream from last instruction start
            init_video_seek = int(gt_instruction_start_time * args.video_fps)
            video_buffer_start_index = init_video_seek
            video_buffer_seek_index = init_video_seek + args.video_seek_amount
        else:
            curr_video_predictions_to_save["pred_texts"].append("Instruction: " + gt_instruction)
            curr_video_predictions_to_save["pred_timestamps"].append(
                data["video_frame_timestamps"][video_buffer_seek_index]
            )

            # continue to stream video
            init_video_seek = video_buffer_seek_index - args.video_seek_amount
            video_buffer_start_index = init_video_seek
            video_buffer_seek_index = init_video_seek + args.video_seek_amount

        while video_buffer_seek_index < len(data['video_frame_paths']):
            logger.info("-" * 40)

            # Optional Policy-A timeout: if the model has failed to signal YES
            # by gt_instruction_end_time + DETECTION_WINDOW_LENGTH, abandon this
            # instruction and let the outer loop advance to the next one.
            # Off by default — opt in via --instruction_timeout.
            if args.instruction_timeout:
                DETECTION_WINDOW_LENGTH = args.detection_time_diff_threshold
                curr_video_time = data["video_frame_timestamps"][video_buffer_seek_index]
                if curr_video_time > gt_instruction_end_time + DETECTION_WINDOW_LENGTH:
                    logger.info(
                        f"[instruction_timeout] curr_video_time={curr_video_time:.1f}s > "
                        f"gt_end({gt_instruction_end_time:.1f}s) + window({DETECTION_WINDOW_LENGTH:.1f}s); "
                        f"abandoning instruction '{gt_instruction}' and advancing"
                    )
                    action_idx += 1
                    if not args.turn_based:
                        # streaming: keep video position, just bump seek to avoid re-entering
                        video_buffer_seek_index += args.video_seek_amount
                        pbar.update(args.video_seek_amount)
                    # turn-based: outer loop will reset seek to next gt_start
                    break

            # cap buffer length to avoid exceeding max_buffer_size
            video_buffer_start_index = max(
                video_buffer_seek_index - args.max_buffer_size,
                init_video_seek,
            )

            logger.info(
                f"{gt_instruction}; "
                f"GT segment end time: {gt_instruction_end_time:.1f}; "
                f"GT segment has mistake:{gt_is_mistake}; "
                f"GT_mistakes: {gt_mistakes_compiled}"
            )
            logger.info(
                f"Curr input clip start to end: {(video_buffer_start_index / 2.)} sec - {(video_buffer_seek_index / 2.)} sec -- "
                f"Curr input clip len: {(video_buffer_seek_index - video_buffer_start_index) / 2.} sec / total len: {(len(data['video_frame_paths']) / 2.)} sec ."
            )

            # check if the person has completed current instruction
            pred_instruction_end = get_qwen_vl_output(
                "instruction_end",
                model,
                processor,
                video=video[video_buffer_start_index:video_buffer_seek_index],
                instruction=gt_instruction,
            )

            logger.info(f"Has the instruction been completed: {pred_instruction_end}")

            if "yes" in pred_instruction_end.lower():
                # move on to next instruction
                pred_instruction_end_time = (
                    data["video_frame_timestamps"][video_buffer_seek_index]
                    - data["video_frame_timestamps"][0]
                )
                logger.info(
                    f"gt_instruction_end_time:{gt_instruction_end_time:.1f}, "
                    f"pred_instruction_end_time:{pred_instruction_end_time:.1f}"
                )

                curr_video_predictions_to_save["pred_texts"].append("Success: Good job!")
                curr_video_predictions_to_save["pred_timestamps"].append(
                    data["video_frame_timestamps"][video_buffer_seek_index]
                )

                action_idx += 1
                video_buffer_seek_index += args.video_seek_amount
                pbar.update(args.video_seek_amount)
                break
            else:
                # If --max_feedback_per_segment is set and we've already hit the cap
                # for this segment, skip the (expensive) mistake_inference call and
                # just slide the window forward. instruction_end at the top of the
                # outer iteration still runs, so Success / timeout detection is
                # unaffected. Default (None) preserves original behavior.
                max_fb = getattr(args, "max_feedback_per_segment", None)
                if max_fb is not None and feedbacks_emitted_this_segment >= max_fb:
                    logger.info(
                        f"[max_feedback_per_segment] cap={max_fb} reached; "
                        f"skipping mistake_inference for this window"
                    )
                    video_buffer_seek_index += args.video_seek_amount
                    pbar.update(args.video_seek_amount)
                    continue

                # --- MCQ branch ---
                # When --mcq_mistake_inference is ON AND a mistake summary exists
                # for the current instruction, force a multiple-choice selection
                # over the listed mistakes (constrained generation, ~4 tokens).
                # Selected text is the summary item verbatim (no free-form gen,
                # no hallucinated wording). Falls through to the standard mistake_
                # inference branch if MCQ is OFF, summary missing, or summary
                # couldn't be parsed into items.
                mcq_summary = _MISTAKE_SUMMARIES.get(gt_instruction) if _MISTAKE_SUMMARIES else None
                mcq_items = (
                    parse_summary_items(mcq_summary)
                    if (getattr(args, "mcq_mistake_inference", False) and mcq_summary)
                    else []
                )
                if mcq_items:
                    mcq_options = "\n".join(f"{n}. {t}" for n, t in mcq_items)
                    mcq_response = get_qwen_vl_output(
                        "mistake_inference_mcq",
                        model,
                        processor,
                        video=video[video_buffer_start_index:video_buffer_seek_index],
                        instruction=gt_instruction,
                        mcq_options=mcq_options,
                        max_new_tokens=4,
                    )
                    logger.info(f"mcq_response: {mcq_response!r}")
                    picked = parse_mcq_response(mcq_response, mcq_items)
                    if picked:
                        predicted_feedback = picked
                        # text-dedup (same rules as the regular path)
                        if getattr(args, "dedupe_feedback_text", False):
                            normalized = " ".join(predicted_feedback.split())
                            if normalized in emitted_feedback_texts_this_segment:
                                logger.info(
                                    f"[dedupe_feedback_text] duplicate dropped (mcq): "
                                    f"{predicted_feedback}"
                                )
                                video_buffer_seek_index += args.video_seek_amount
                                pbar.update(args.video_seek_amount)
                                continue
                            emitted_feedback_texts_this_segment.add(normalized)

                        # Grounding DINO post-hoc verification.
                        # Look up the pre-extracted object nouns for this summary
                        # item; if none detected in the last frame of the current
                        # window, drop the emit (likely hallucination FP). Empty
                        # object list → keep (conservative).
                        if verifier is not None:
                            expected_objects = lookup_objects(predicted_feedback)
                            if expected_objects:
                                last_frame_idx = max(video_buffer_seek_index - 1, 0)
                                last_frame_path = data["video_frame_paths"][last_frame_idx]

                                # Build optional viz path: <viz_dir>/<video_id>/seg{idx}_t{ts}s.jpg
                                viz_path = None
                                if getattr(args, "grounding_dino_save_viz_dir", None):
                                    ts = data["video_frame_timestamps"][last_frame_idx]
                                    viz_path = os.path.join(
                                        args.grounding_dino_save_viz_dir,
                                        str(data["video_id"]),
                                        f"seg{action_idx:03d}_t{ts:07.1f}s.jpg",
                                    )

                                vr = verifier.verify_detailed(
                                    last_frame_path,
                                    expected_objects,
                                    save_viz_path=viz_path,
                                    feedback_text=predicted_feedback,
                                )

                                # Verbose per-query logging
                                if getattr(args, "grounding_dino_verbose", False):
                                    logger.info(
                                        f"[grounding_dino] frame={last_frame_path} "
                                        f"objects={expected_objects} "
                                        f"feedback={predicted_feedback!r}"
                                    )
                                    for q in vr["per_query"]:
                                        # top-3 each to keep lines short
                                        top = list(zip(q["labels"][:3], q["scores"][:3], q["boxes"][:3]))
                                        logger.info(
                                            f"[grounding_dino]   q={q['query']!r} "
                                            f"n_dets={q['n_dets']} top={top}"
                                        )

                                if not vr["decision"]:
                                    logger.info(
                                        f"[grounding_dino] DROP frame={last_frame_path} "
                                        f"objects={expected_objects} feedback={predicted_feedback!r}"
                                    )
                                    # NOTE: don't un-add from emitted_feedback_texts set —
                                    # keeping it prevents re-running verify on the same text
                                    # in this segment (it would just keep dropping).
                                    video_buffer_seek_index += args.video_seek_amount
                                    pbar.update(args.video_seek_amount)
                                    continue
                                logger.info(
                                    f"[grounding_dino] KEEP frame={last_frame_path} "
                                    f"objects={expected_objects} feedback={predicted_feedback!r}"
                                )

                        logger.info(f"Saving feedback (mcq): -- {predicted_feedback}")
                        curr_video_predictions_to_save["pred_texts"].append(
                            f"Feedback: {predicted_feedback}"
                        )
                        curr_video_predictions_to_save["pred_timestamps"].append(
                            data["video_frame_timestamps"][video_buffer_seek_index]
                        )
                        feedbacks_emitted_this_segment += 1
                    video_buffer_seek_index += args.video_seek_amount
                    pbar.update(args.video_seek_amount)
                    continue
                # --- end MCQ branch ---

                # check if person has made a mistake
                # If a per-instruction mistake summary is available, use the
                # variant prompt that injects it (replacing the generic 5-category
                # error list). Otherwise fall back to the original prompt.
                summary = _MISTAKE_SUMMARIES.get(gt_instruction) if _MISTAKE_SUMMARIES else None
                if summary:
                    mistake_inference = get_qwen_vl_output(
                        "mistake_inference_with_summary",
                        model,
                        processor,
                        video=video[video_buffer_start_index:video_buffer_seek_index],
                        instruction=gt_instruction,
                        mistake_summary=summary,
                        max_new_tokens=1024,
                    )
                else:
                    mistake_inference = get_qwen_vl_output(
                        "mistake_inference",
                        model,
                        processor,
                        video=video[video_buffer_start_index:video_buffer_seek_index],
                        instruction=gt_instruction,
                        max_new_tokens=1024,
                    )

                logger.info(f"mistake_inference: {mistake_inference}")

                if "yes" in mistake_inference.lower():
                    predicted_feedback = (
                        mistake_inference.lower().replace("yes.", "").replace("yes", "").strip()
                    )

                    # Optional --dedupe_feedback_text: if this exact normalized text
                    # was already saved earlier in this segment, drop the emit (and
                    # do NOT count it against --max_feedback_per_segment so distinct
                    # feedbacks still get their full budget).
                    if getattr(args, "dedupe_feedback_text", False):
                        normalized = " ".join(predicted_feedback.split())
                        if normalized in emitted_feedback_texts_this_segment:
                            logger.info(
                                f"[dedupe_feedback_text] duplicate dropped: {predicted_feedback}"
                            )
                            video_buffer_seek_index += args.video_seek_amount
                            pbar.update(args.video_seek_amount)
                            continue
                        emitted_feedback_texts_this_segment.add(normalized)

                    logger.info(f"Saving feedback: --  {predicted_feedback}")
                    curr_video_predictions_to_save["pred_texts"].append(
                        f"Feedback: {predicted_feedback}"
                    )
                    curr_video_predictions_to_save["pred_timestamps"].append(
                        data["video_frame_timestamps"][video_buffer_seek_index]
                    )
                    feedbacks_emitted_this_segment += 1

                video_buffer_seek_index += args.video_seek_amount
                pbar.update(args.video_seek_amount)

    pbar.close()
    return curr_video_predictions_to_save


def _atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON atomically (tmp file + rename) so crashes don't leave a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _load_resume_state(save_file_path: str, logger) -> List[Dict[str, Any]]:
    """Return previously-saved predictions if ``save_file_path`` exists and is valid, else []."""
    if not os.path.exists(save_file_path):
        return []
    try:
        with open(save_file_path) as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            logger.warning(f"[resume] {save_file_path} is not a list; starting fresh")
            return []
        # Sanity-check each entry has a video_id
        existing = [p for p in existing if isinstance(p, dict) and "video_id" in p]
        return existing
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[resume] Could not load {save_file_path}: {e}; starting fresh")
        return []


def _run_inference_loop(
    args,
    eval_idxs,
    save_file_path,
    log_file_path,
    device_map,
    worker_id: Optional[int] = None,
    gpu_id: Optional[int] = None,
):
    """Set up logger + dataset + model and iterate through ``eval_idxs`` in this process.

    When ``worker_id`` is provided (multi-process mode), the logger writes
    file-only and the outer tqdm bar is pinned to row ``worker_id`` so multiple
    workers stack cleanly in the shared terminal.

    When ``args.resume`` is True and ``save_file_path`` already contains valid
    predictions, video_ids found there are skipped — execution continues only
    on the remaining videos.
    """
    if args.prompt_config_file is not None:
        load_prompt_configs_from_file(args.prompt_config_file)

    if getattr(args, "mistake_summary_file", None):
        load_mistake_summaries(args.mistake_summary_file)

    if getattr(args, "mistake_objects_file", None):
        load_mistake_objects(args.mistake_objects_file)

    # Initialize Grounding DINO verifier (one per worker process).
    # Stays None if --grounding_dino_verify is OFF.
    verifier = None
    if getattr(args, "grounding_dino_verify", False):
        from tools.grounding_dino_verifier import GroundingDinoVerifier
        verifier = GroundingDinoVerifier(
            model_id=args.grounding_dino_model_id,
            device=device_map if device_map.startswith("cuda") else "cuda:0",
            cache_dir=args.cache_dir,
            box_threshold=args.grounding_dino_box_threshold,
        )

    video_input_resolution = (args.video_input_width, args.video_input_height)

    is_worker = worker_id is not None
    log_mode = "a" if getattr(args, "resume", False) and os.path.exists(log_file_path) else "w"
    logger = _setup_logger(log_file_path, also_stdout=not is_worker, mode=log_mode)
    logger.info(f"Saving predictions to {save_file_path}")
    logger.info(f"Saving logs to {log_file_path}")
    logger.info(f"Processing {len(eval_idxs)} videos | device_map={device_map}")

    dataset = QualcommInteractiveCookingDatasetVideos(
        captaincook4d_root=args.captaincook4d_root,
        plan_set=args.plan_set,
        split=args.split,
        model_fps=args.model_fps,
    )

    # Resume: drop indices whose video_id was already completed in a prior run.
    predictions_to_save: List[Dict[str, Any]] = []
    if getattr(args, "resume", False):
        existing = _load_resume_state(save_file_path, logger)
        if existing:
            completed_ids = {p["video_id"] for p in existing}
            remaining = [i for i in eval_idxs if dataset[i]["video_id"] not in completed_ids]
            skipped = len(eval_idxs) - len(remaining)
            if skipped > 0:
                logger.info(
                    f"[resume] {skipped}/{len(eval_idxs)} videos already done; "
                    f"continuing with {len(remaining)} remaining"
                )
                predictions_to_save = existing
                eval_idxs = remaining

    if is_worker:
        outer_desc = f"W{worker_id} (gpu{gpu_id})" if gpu_id is not None else f"W{worker_id}"
        outer_pbar = tqdm(eval_idxs, desc=outer_desc, position=worker_id, leave=True)
        inner_pbar_enabled = False
    else:
        outer_pbar = tqdm(eval_idxs, desc="videos")
        inner_pbar_enabled = True

    # Load model only if there's actually work left to do.
    if len(eval_idxs) == 0:
        logger.info("[resume] No remaining videos for this worker; skipping model load.")
        return

    model, processor = load_model_and_processor(args.model_id, args.cache_dir, device_map)

    for eval_idx in outer_pbar:
        logger.info("=" * 80)
        data = dataset[eval_idx]
        video = load_frames_into_array(
            data['video_frame_paths'],
            video_input_resolution=video_input_resolution,
        )
        result = process_one_video(
            eval_idx, data, video, model, processor, args, logger,
            inner_pbar=inner_pbar_enabled,
            verifier=verifier,
        )
        predictions_to_save.append(result)
        _atomic_write_json(save_file_path, predictions_to_save)
        logger.info(">>>\n")


def _worker_entry(worker_id, gpu_id, video_indices, args, save_path, log_path):
    """Subprocess entry-point. Pin this worker's default CUDA device to ``gpu_id``.

    NOTE: We intentionally do NOT override CUDA_VISIBLE_DEVICES inside the
    worker. ``multiprocessing`` spawn re-imports the main module before this
    function runs, which can trigger transitive imports that initialise the
    CUDA runtime against the *parent's* CUDA_VISIBLE_DEVICES. Once CUDA is
    initialised, later changes to the env var are ignored, causing every
    worker to silently land on the same physical GPU.

    Instead we set the default device explicitly and load the model with
    ``device_map=f"cuda:{gpu_id}"``. ``gpu_id`` follows PyTorch convention:
    it is an index into the GPUs visible to this process (i.e. relative to
    the launcher's CUDA_VISIBLE_DEVICES if set, otherwise absolute physical
    IDs).
    """
    torch.cuda.set_device(gpu_id)
    _run_inference_loop(
        args, video_indices, save_path, log_path, device_map=f"cuda:{gpu_id}",
        worker_id=worker_id, gpu_id=gpu_id,
    )


def run(args):
    """
    Main evaluation entrypoint.

    Dispatches to either:
      - single-process inference (default; preserves the original behavior), or
      - multi-process inference across one or more GPUs and one or more model
        copies per GPU, when ``--gpus`` and/or ``--workers_per_gpu`` are set.

    In multi-process mode each worker writes to ``part<i>_<save_file>`` and the
    main process merges them into ``<save_file>`` at the end.
    """
    # ---- Decide output directory layout ----
    # If --save_exp is given (e.g. "qwen3vl_4b_timeout"):
    #   predictions/<save_exp>/<save_exp>.json                <- merged result
    #   predictions/<save_exp>/part<i>_<save_exp>.json        <- per-worker
    #   log/<save_exp>/<save_exp>_log.txt                     <- single-process log
    #   log/<save_exp>/part<i>_<save_exp>_log.txt             <- per-worker logs
    # Otherwise (default), keep the legacy flat layout under save_root / log_root.
    parent_dir = os.path.dirname(args.save_root) or "."
    if args.save_exp:
        file_name = f"{args.save_exp}.json"
        pred_dir = os.path.join(args.save_root, args.save_exp)
        log_dir = os.path.join(parent_dir, "log", args.save_exp)
        log_base = args.save_exp
    else:
        file_name = args.save_file
        pred_dir = args.save_root
        log_dir = os.path.join(parent_dir, "log")
        log_base = os.path.splitext(args.save_file)[0]

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    save_file_path = os.path.join(pred_dir, file_name)

    # ---- Parse parallelism config ----
    gpu_ids = [int(g.strip()) for g in str(args.gpus).split(",") if g.strip() != ""]
    if not gpu_ids:
        gpu_ids = [0]
    workers_per_gpu = max(1, int(args.workers_per_gpu))
    total_workers = len(gpu_ids) * workers_per_gpu

    # ---- Single-process path (backward-compatible default) ----
    if total_workers == 1:
        dataset_for_count = QualcommInteractiveCookingDatasetVideos(
            captaincook4d_root=args.captaincook4d_root,
            plan_set=args.plan_set,
            split=args.split,
            model_fps=args.model_fps,
        )
        eval_idxs = list(range(len(dataset_for_count)))
        del dataset_for_count

        log_file_path = os.path.join(log_dir, f"{log_base}_log.txt")
        _run_inference_loop(
            args, eval_idxs, save_file_path, log_file_path, args.device_map
        )
        return

    # ---- Multi-process path ----
    print(
        f"[run] Multi-process mode: {len(gpu_ids)} GPU(s) "
        f"x {workers_per_gpu} workers/GPU = {total_workers} workers total | gpus={gpu_ids}"
    )

    tmp_ds = QualcommInteractiveCookingDatasetVideos(
        captaincook4d_root=args.captaincook4d_root,
        plan_set=args.plan_set,
        split=args.split,
        model_fps=args.model_fps,
    )
    n_videos = len(tmp_ds)
    del tmp_ds

    # Round-robin GPU assignment: gpu0-w0, gpu0-w1, ..., gpu1-w0, gpu1-w1, ...
    worker_gpus: List[int] = []
    for g in gpu_ids:
        for _ in range(workers_per_gpu):
            worker_gpus.append(g)

    # Interleave video indices for load balance (avoids clustering long videos in one worker)
    indices_per_worker: List[List[int]] = [[] for _ in range(total_workers)]
    for i in range(n_videos):
        indices_per_worker[i % total_workers].append(i)

    print(
        f"[run] {n_videos} videos -> per-worker counts: "
        f"{[len(x) for x in indices_per_worker]}"
    )

    # CUDA requires 'spawn' for sub-processes
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    procs: List[mp.Process] = []
    save_paths: List[str] = []
    for w_id in range(total_workers):
        wsave = os.path.join(pred_dir, f"part{w_id}_{file_name}")
        wlog = os.path.join(log_dir, f"part{w_id}_{log_base}_log.txt")
        save_paths.append(wsave)
        p = mp.Process(
            target=_worker_entry,
            args=(w_id, worker_gpus[w_id], indices_per_worker[w_id], args, wsave, wlog),
        )
        p.start()
        procs.append(p)
        print(
            f"[run] launched worker {w_id}: pid={p.pid}, gpu={worker_gpus[w_id]}, "
            f"n_videos={len(indices_per_worker[w_id])}, log={wlog}"
        )

    for p in procs:
        p.join()

    failed = [(i, p.exitcode) for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        print(f"[run] WARNING: {len(failed)} worker(s) exited non-zero: {failed}")

    # ---- Merge partial files ----
    # Dedupe by video_id in case a previous run with different worker assignment
    # left predictions for the same video in two part files (resume scenario).
    all_preds: List[Dict[str, Any]] = []
    seen_video_ids = set()
    for sp in save_paths:
        if not os.path.exists(sp):
            print(f"[run] WARNING: missing partial file {sp}")
            continue
        with open(sp) as f:
            for p in json.load(f):
                vid = p.get("video_id")
                if vid in seen_video_ids:
                    continue
                seen_video_ids.add(vid)
                all_preds.append(p)

    _atomic_write_json(save_file_path, all_preds)

    print(f"[run] merged {len(all_preds)} predictions -> {save_file_path}")
    print(
        f"[run] partial files kept at {pred_dir}/part*_{file_name} "
        f"(delete after sanity check if not needed)"
    )


def parse_args():
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Run cooking video evaluation with Qwen3-VL")

    parser.add_argument("--turn_based", action="store_true", default=False,
                        help="Whether to use turn-based mode")
    parser.add_argument("--instruction_timeout", action="store_true", default=False,
                        help=(
                            "If set, abandon the current instruction segment once the video "
                            "time exceeds gt_instruction_end_time + DETECTION_WINDOW_LENGTH "
                            "(=--detection_time_diff_threshold, default 15.0s) without the "
                            "model signaling YES. Prevents the loop from getting stuck on one "
                            "instruction when the model never confirms completion."
                        ))
    parser.add_argument("--resume", action="store_true", default=False,
                        help=(
                            "Resume a crashed/interrupted run. Each worker reads its own "
                            "part<i>_<exp>.json (or the single-process file) and skips video_ids "
                            "already present, continuing only on the remaining videos. "
                            "Saves are atomic (tmp + rename) so the on-disk file is always "
                            "well-formed even after a crash."
                        ))
    parser.add_argument("--dataset_name", type=str, default="eccv",
                        help="Dataset name")
    parser.add_argument("--save_root", type=str, default="./predictions",
                        help="Root directory to save predictions")
    parser.add_argument("--save_file", type=str, default="predictions.json",
                        help="Filename to save predictions. Ignored if --save_exp is set.")
    parser.add_argument("--save_exp", type=str, default=None,
                        help="Experiment name. When set (e.g. 'qwen3vl_4b_timeout'), "
                             "outputs are written under <save_root>/<save_exp>/ and "
                             "<log_root>/<save_exp>/ with <save_exp>.json as the basename. "
                             "Overrides --save_file's basename.")
    parser.add_argument("--captaincook4d_root", type=str, required=True,
                        help="Root directory of the CaptainCook4D dataset (contains <video_id>_360p.mp4 files).")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--plan_set", type=str, default="main",
                        help="Annotation type")
    parser.add_argument("--model_fps", type=int, default=2,
                        help="Model FPS")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model ID to load from HuggingFace")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for model weights")
    parser.add_argument("--llm_max_new_tokens", type=int, default=128,
                        help="Maximum new tokens for LLM generation")
    parser.add_argument("--max_buffer_size", type=int, default=240,
                        help="Maximum video buffer size in frames (2 minutes at 2fps)")
    parser.add_argument("--video_fps", type=int, default=2,
                        help="Video frames per second")
    parser.add_argument("--detection_time_diff_threshold", type=float, default=15.0,
                        help="Detection time difference threshold in frames (30 sec / 2 fps)")
    parser.add_argument("--video_seek_amount", type=int, default=10,
                        help="Number of frames to seek forward in video")
    parser.add_argument("--prompt_config_file", type=str, default=None,
                        help=(
                            "Optional path to a JSON file that overrides the default prompt configs. "
                            "Expected format: {\"<mode>\": {\"system_prompt\": \"...\", \"user_text_template\": \"...\"}, ...}"
                        ))
    parser.add_argument("--mistake_summary_file", type=str, default=None,
                        help=(
                            "Optional path to a JSON of {instruction_text: mistake_summary_text} "
                            "produced by tools/build_mistake_summaries.py. When set, the "
                            "mistake-check call uses the 'mistake_inference_with_summary' "
                            "prompt (paper-style per-instruction summary) for any instruction "
                            "with a cached summary, and falls back to the generic 5-category "
                            "'mistake_inference' prompt for instructions without one."
                        ))
    parser.add_argument("--max_feedback_per_segment", type=int, default=None,
                        help=(
                            "Cap the number of 'Feedback:' predictions emitted within a "
                            "single instruction segment. Once the cap is reached, further "
                            "mistake_inference calls in that segment are skipped (the "
                            "instruction_end check still fires, so segment completion / "
                            "timeout still works). Default None -> unlimited (original "
                            "behavior, fully back-compat). Set to 1 to suppress the "
                            "repeated-feedback-per-window pattern that inflates FP: "
                            "the evaluator counts at most one TP per GT mistake but treats "
                            "every Feedback as an independent FP candidate, so a segment "
                            "firing the same feedback 10 times yields 1 TP + 9 FP. "
                            "Composes with --mistake_summary_file and --instruction_timeout."
                        ))
    parser.add_argument("--dedupe_feedback_text", action="store_true", default=False,
                        help=(
                            "Suppress emission of a Feedback whose normalized text "
                            "(lowercased, whitespace-collapsed) was already saved earlier "
                            "in the same instruction segment. Default OFF (original "
                            "behavior). Targets the residual same-meaning-different-wording "
                            "FP pattern that --max_feedback_per_segment doesn't always "
                            "catch. Composes with --max_feedback_per_segment: dedup is "
                            "checked AFTER the mistake_inference call, and only distinct "
                            "(non-duplicate) feedbacks count against the cap counter."
                        ))
    parser.add_argument("--mcq_mistake_inference", action="store_true", default=False,
                        help=(
                            "Use multiple-choice (MCQ) mistake detection instead of "
                            "free-form generation. For each instruction with a cached "
                            "mistake summary, the model is shown the numbered summary "
                            "items + 'N' (none) and must answer with ONE character. The "
                            "picked summary item is emitted VERBATIM as the Feedback "
                            "text -- no free-form generation, no hallucinated wording. "
                            "Default OFF. Requires --mistake_summary_file; silently "
                            "falls back to the standard mistake_inference path for "
                            "instructions without a summary. Composes orthogonally with "
                            "--max_feedback_per_segment, --dedupe_feedback_text, "
                            "--instruction_timeout, and --prompt_config_file (which can "
                            "override the 'mistake_inference_mcq' prompt). Targets "
                            "hallucination-FP by constraining the answer space and also "
                            "improves BERT/ROUGE since emit text matches the curated "
                            "summary."
                        ))
    parser.add_argument("--grounding_dino_verify", action="store_true", default=False,
                        help=(
                            "Post-hoc visual grounding verification: after the VLM "
                            "(MCQ path) emits a Feedback, look up the pre-extracted "
                            "object nouns for that summary item and verify their "
                            "presence in the last frame of the current window using "
                            "Grounding DINO. Drop the emit if NO listed object is "
                            "detected (likely hallucination FP). Empty/missing object "
                            "list -> keep (conservative). Requires "
                            "--mistake_objects_file. Currently only affects the MCQ "
                            "path (emit text is a verbatim summary item, looked up "
                            "directly). Composes with all other flags."
                        ))
    parser.add_argument("--mistake_objects_file", type=str, default=None,
                        help=(
                            "Path to {summary_item_text: [object_noun, ...]} JSON "
                            "produced by tools/extract_summary_objects.py. Required "
                            "with --grounding_dino_verify."
                        ))
    parser.add_argument("--grounding_dino_model_id", type=str,
                        default="IDEA-Research/grounding-dino-tiny",
                        help=(
                            "HF model id for Grounding DINO. Default 'tiny' (~170MB) "
                            "for speed. Use 'IDEA-Research/grounding-dino-base' for "
                            "stronger detection (~700MB) at higher latency/VRAM."
                        ))
    parser.add_argument("--grounding_dino_box_threshold", type=float, default=0.4,
                        help=(
                            "DINO detection score threshold. Default 0.4. Empirical "
                            "test: at 0.3 DINO sometimes produces spurious detections "
                            "for strong-prior text queries even on featureless frames. "
                            "0.4-0.5 is safer; 0.6+ is strict. Tune based on log "
                            "[grounding_dino] DROP/KEEP rates."
                        ))
    parser.add_argument("--grounding_dino_verbose", action="store_true", default=False,
                        help=(
                            "Log per-query DINO detection details (n_dets, top scores, "
                            "labels, boxes) for every verify call. Useful for tuning "
                            "the threshold or auditing what objects DINO is actually "
                            "seeing. Roughly doubles the [grounding_dino] log volume."
                        ))
    parser.add_argument("--grounding_dino_save_viz_dir", type=str, default=None,
                        help=(
                            "If set, save a bbox-overlay JPG for every verify call to "
                            "this directory (one file per emit, grouped by video_id). "
                            "Header shows KEEP/DROP + feedback text; boxes are colored "
                            "per query with score/label. Disk-heavy; use only for "
                            "exploratory runs. Default None (off)."
                        ))
    parser.add_argument("--device_map", type=str, default="cuda:0",
                        help="Device map for model loading (e.g. 'cuda:0', 'auto'). "
                             "Used only in single-process mode (--gpus '0' --workers_per_gpu 1).")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU indices for multi-process inference. "
                             "PyTorch convention: relative to the launcher's CUDA_VISIBLE_DEVICES "
                             "when set, otherwise absolute physical IDs. "
                             "Default '0'. Example: '0,1,2,3'. "
                             "If you launch with CUDA_VISIBLE_DEVICES=6,7, use --gpus '0,1' to "
                             "target those two GPUs.")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Number of parallel model copies per GPU. "
                             "Total workers = len(gpus) * workers_per_gpu. "
                             "Default 1 -> single-process when --gpus is '0' (original behavior). "
                             "On high-VRAM GPUs (H100/H200), 2-4 copies of a small model can saturate compute.")
    parser.add_argument("--video_input_width", type=int, default=640,
                        help="Width of the video input resolution")
    parser.add_argument("--video_input_height", type=int, default=360,
                        help="Height of the video input resolution")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
