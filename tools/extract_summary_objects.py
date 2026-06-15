"""
extract_summary_objects.py — offline extraction of physical object nouns
from per-instruction mistake summary items.

For each summary item (one line from the numbered list produced by
``build_mistake_summaries.py``), use a small instruction-tuned LLM to extract
the concrete physical objects mentioned (lowercase singular nouns). The
output is consumed at inference time by ``--grounding_dino_verify`` in
``inference.py`` to verify VLM mistake claims against a visual grounding
detector (Grounding DINO).

Output JSON shape:
    {
      "<summary item text, verbatim>": ["spoon", "knife", "tortilla"],
      ...
    }

The key is the summary item text VERBATIM (no trailing period stripped, no
case change) since at inference time the MCQ path emits the summary item as
the Feedback string and looks it up directly. We also store a second copy
keyed by the whitespace-normalized text so the lookup is robust to minor
whitespace drift.

Usage
-----
    python tools/extract_summary_objects.py \\
        --summaries_file assets/mistake_summaries_main_test.json \\
        --output_path assets/mistake_objects_main_test.json

    # Resume an interrupted run:
    python tools/extract_summary_objects.py ... --resume

Dependencies
------------
    (transformers, torch already in the project env)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


EXTRACTOR_PROMPT_TEMPLATE_V1 = (
    "Extract the concrete physical objects mentioned in this cooking mistake "
    "description. Objects are tools, utensils, ingredients, containers, or "
    "appliances that would be visible in a kitchen camera frame.\n\n"
    "Rules:\n"
    "- Output a JSON array of singular lowercase nouns (or short noun phrases).\n"
    "- Skip abstract concepts (e.g. 'mistake', 'amount', 'time', 'temperature').\n"
    "- Skip body parts (e.g. 'hand', 'finger').\n"
    "- Compound nouns are OK: 'butter knife', 'nut butter', 'cutting board'.\n"
    "- If no concrete object is mentioned, output an empty array [].\n\n"
    "Description: \"Users might use a spoon instead of a butter knife to scoop nut butter.\"\n"
    "Output: [\"spoon\", \"butter knife\", \"nut butter\"]\n\n"
    "Description: \"Users may not preheat the pan long enough.\"\n"
    "Output: [\"pan\"]\n\n"
    "Description: \"Users could time the saute incorrectly.\"\n"
    "Output: []\n\n"
    "Description: \"{item}\"\n"
    "Output:"
)


# V2 — DINO-friendly canonical objects only.
# Motivated by viz inspection of qwen3vl_8b_..._gdino run: V1 was extracting
# parts ('handle', 'blade'), state words ('log' for rolled tortilla), verbs
# ('chopping'), and attributes ('papery skin') that Grounding DINO cannot
# reliably ground in a kitchen camera frame. V2 tightens the rules to only
# WHOLE, COMMONLY-NAMED, VISIBLE kitchen items so DINO has a fair shot.
EXTRACTOR_PROMPT_TEMPLATE_V2 = (
    "Extract the WHOLE, VISIBLE, CANONICALLY-NAMED kitchen objects mentioned in "
    "this cooking mistake description. The output will be used as text queries "
    "for a generic visual object detector (Grounding DINO), so only include "
    "objects that a non-cooking-expert would recognize at a glance.\n\n"
    "STRICT rules:\n"
    "- Output a JSON array of singular lowercase nouns (or short noun phrases).\n"
    "- Include WHOLE objects only:\n"
    "    OK   -> 'knife', 'pan', 'tortilla', 'onion', 'garlic', 'measuring spoon'\n"
    "    SKIP -> 'handle', 'blade', 'edge', 'lid' (parts of objects)\n"
    "    SKIP -> 'log' (state/shape of a rolled tortilla, not a canonical object)\n"
    "    SKIP -> 'papery skin', 'root end' (attributes / sub-parts)\n"
    "- Skip ACTIONS / VERBS: 'chopping', 'stirring', 'pouring', 'scoop' (the action).\n"
    "  Keep the NOUN: 'spoon' (the tool) is OK.\n"
    "- Skip ABSTRACT concepts: 'mistake', 'amount', 'time', 'temperature', "
    "'consistency', 'texture'.\n"
    "- Skip BODY parts: 'hand', 'finger'.\n"
    "- Skip QUANTITIES: '1/2 teaspoon' (just include the tool/ingredient if any).\n"
    "- Compound nouns for common kitchen items are OK and PREFERRED when more "
    "specific: 'butter knife', 'cutting board', 'measuring spoon', 'nut butter'.\n"
    "- If the description has no concrete WHOLE-object reference, output [].\n\n"
    "Description: \"Users might use a spoon instead of a butter knife to scoop nut butter.\"\n"
    "Output: [\"spoon\", \"butter knife\", \"nut butter\"]\n\n"
    "Description: \"Users may not preheat the pan long enough.\"\n"
    "Output: [\"pan\"]\n\n"
    "Description: \"Users could time the saute incorrectly.\"\n"
    "Output: []\n\n"
    "Description: \"You are only wiping the handle and not the blade surface.\"\n"
    "Output: [\"knife\"]\n\n"
    "Description: \"You rolled the tortilla too loosely, leaving visible gaps in the log.\"\n"
    "Output: [\"tortilla\"]\n\n"
    "Description: \"You left the papery skin on one of the garlic cloves.\"\n"
    "Output: [\"garlic\"]\n\n"
    "Description: \"You didn't remove the root end before chopping.\"\n"
    "Output: [\"onion\"]\n\n"
    "Description: \"You used a knife to slice the onion instead of chopping it.\"\n"
    "Output: [\"knife\", \"onion\"]\n\n"
    "Description: \"{item}\"\n"
    "Output:"
)


# Registry — pick by --prompt_version
PROMPT_VERSIONS: Dict[str, str] = {
    "v1": EXTRACTOR_PROMPT_TEMPLATE_V1,
    "v2": EXTRACTOR_PROMPT_TEMPLATE_V2,
}


def parse_summary_items(summary_text: str) -> List[str]:
    """Same parser as in inference.py — returns verbatim item texts."""
    items: List[str] = []
    for line in summary_text.splitlines():
        m = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", line)
        if m:
            items.append(m.group(2).strip())
    return items


def parse_extractor_output(text: str) -> List[str]:
    """Find the first JSON array of strings in the LLM output."""
    # Try to find [ ... ] block
    m = re.search(r"\[(.*?)\]", text, flags=re.DOTALL)
    if not m:
        return []
    body = m.group(0)
    try:
        arr = json.loads(body)
        if isinstance(arr, list):
            return [str(x).strip().lower() for x in arr if str(x).strip()]
    except json.JSONDecodeError:
        pass
    # Fallback: split on commas/quotes
    inner = m.group(1)
    parts = re.findall(r'"([^"]+)"', inner)
    return [p.strip().lower() for p in parts if p.strip()]


@torch.no_grad()
def extract_one(item: str, tokenizer, model, prompt_template: str, max_new_tokens: int = 64) -> List[str]:
    prompt = prompt_template.format(item=item)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return parse_extractor_output(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract object nouns from mistake summary items.")
    p.add_argument("--summaries_file", required=True,
                   help="Path to {instruction: summary_text} JSON from build_mistake_summaries.py.")
    p.add_argument("--output_path", required=True,
                   help="Where to save {item_text: [objects]} JSON.")
    p.add_argument("--llm_model_id", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                   help="LLM for noun extraction. Same family as build_mistake_summaries.py.")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true",
                   help="Skip items already present in --output_path.")
    p.add_argument("--limit", type=int, default=None,
                   help="(Debug) only process first N unique items.")
    p.add_argument("--prompt_version", choices=list(PROMPT_VERSIONS.keys()),
                   default="v2",
                   help=(
                       "Which extractor prompt to use. v1 = original (looser, "
                       "may produce parts/verbs that Grounding DINO can't match). "
                       "v2 (default) = DINO-friendly canonical objects only, "
                       "explicitly skips parts/verbs/attributes/state words. "
                       "Tuned from viz inspection of the first gdino run."
                   ))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # 1. Load summaries
    print(f"[1/4] Loading summaries: {args.summaries_file}")
    with open(args.summaries_file) as f:
        summaries: Dict[str, str] = json.load(f)
    print(f"      -> {len(summaries)} instructions")

    # 2. Collect unique summary items
    print(f"[2/4] Collecting unique summary items")
    unique_items: set = set()
    for summary in summaries.values():
        for item in parse_summary_items(summary):
            unique_items.add(item)
    items_list = sorted(unique_items)
    print(f"      -> {len(items_list)} unique items")
    if args.limit:
        items_list = items_list[: args.limit]

    # 3. Resume cache
    extracted: Dict[str, List[str]] = {}
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path) as f:
            extracted = json.load(f)
        print(f"[resume] loaded {len(extracted)} existing extractions")
    remaining = [it for it in items_list if it not in extracted]
    print(f"      -> {len(remaining)} items to process")

    if not remaining:
        print("Nothing to do. Exiting.")
        return

    # 4. Load LLM
    print(f"[3/4] Loading extractor LLM: {args.llm_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_id, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_id,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
    ).eval()

    # 5. Extract
    prompt_template = PROMPT_VERSIONS[args.prompt_version]
    print(f"[4/4] Extracting object nouns (prompt_version={args.prompt_version})")
    for item in tqdm(remaining, desc="extracting"):
        objects = extract_one(item, tokenizer, model, prompt_template)
        extracted[item] = objects
        # incremental atomic save
        tmp = args.output_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
        os.replace(tmp, args.output_path)

    print(f"\nDone. Wrote {len(extracted)} extractions -> {args.output_path}")
    # show some samples
    sample_keys = list(extracted.keys())[:5]
    for k in sample_keys:
        print(f"  {k!r}\n    -> {extracted[k]}")


if __name__ == "__main__":
    main()
