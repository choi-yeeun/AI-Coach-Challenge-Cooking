"""
build_mistake_summaries.py — build per-instruction "mistake summary" cache.

For each unique instruction in the target split (e.g., main/test), retrieve the
top-k most similar instructions from the retrieval source split (default: the
train split of the same plan_set), and use an instruction-tuned text LLM to
summarize 3-5 plausible mistakes a user might make on the target instruction,
conditioned on the annotated mistakes of the retrieved similar instructions.

This follows the procedure described in Bhattacharyya et al. (NeurIPS 2025,
arXiv:2511.21998) Appendix F, p.22. We replace their Qwen-2.5-32B-Instruct with
``Qwen/Qwen3-30B-A3B-Instruct-2507`` by default (Qwen3 MoE, ~3B active params,
faster). Override with ``--llm_model_id`` if needed.

Output JSON shape:
    {
      "<instruction text>": "1. <mistake>\\n2. <mistake>\\n3. <mistake>\\n...",
      ...
    }

The output file is consumed by ``inference.py`` via
``--mistake_summary_file <path>``. Without that flag the inference loop keeps
using its original generic 5-category mistake prompt.

Usage
-----
    python tools/build_mistake_summaries.py \\
        --plan_set main --split test \\
        --output_path assets/mistake_summaries_main_test.json

    # If the LLM run is interrupted, re-run with --resume to skip already-summarized
    # instructions.

Dependencies
------------
    pip install sentence-transformers
    (transformers, datasets, torch already in the project env.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


HF_DATASET_NAME = "qualcomm/qualcomm-interactive-cooking-dataset"


SUMMARIZER_USER_TEMPLATE = (
    "You are an expert cooking assistant designing a "
    "mistake-detection checklist for a multi-modal assistant that watches a person cook.\n\n"
    "Given a target recipe instruction and several SIMILAR instructions (drawn from a training "
    "set, each with annotated user mistakes that actually occurred), write 3-5 CONCRETE potential "
    "mistakes a user might make while attempting the TARGET instruction.\n\n"
    "Rules:\n"
    "- Each mistake must be specific to the target instruction (mention the same ingredients, "
    "quantities, tools, or steps where applicable).\n"
    "- Mistakes should be the kind a person could plausibly make on this exact step (e.g. wrong "
    "amount, wrong tool, wrong technique, partial completion).\n"
    "- Each mistake one short sentence.\n"
    "- Output FORMAT — exactly a numbered list, no preamble, no closing remark:\n"
    "  1. <mistake A>\n"
    "  2. <mistake B>\n"
    "  3. <mistake C>\n\n"
    "Similar instructions and their annotated mistakes from training data:\n"
    "{examples}\n\n"
    "Target instruction:\n"
    "{instruction}\n\n"
    "Output (numbered list only):"
)


def load_instruction_mistake_pairs(plan_set: str, split: str) -> List[Tuple[str, List[str]]]:
    """Walk the HF annotations and return [(instruction_text, list_of_mistake_feedbacks), ...].

    A mistake is any output entry whose ``output_types`` contains the substring
    ``"mistake"``. Pairs are grouped by instruction segment (instruction →
    following feedbacks until the next instruction).
    """
    ds = load_dataset(HF_DATASET_NAME, plan_set, split=split)
    pairs: List[Tuple[str, List[str]]] = []
    for row in ds:
        texts = row["output_texts"]
        types = row["output_types"]
        cur_inst: str | None = None
        cur_mistakes: List[str] = []
        for t, ty in zip(texts, types):
            if ty == "instruction":
                if cur_inst is not None:
                    pairs.append((cur_inst, cur_mistakes))
                cur_inst = t
                cur_mistakes = []
            elif "mistake" in ty:
                cur_mistakes.append(t)
            # success/finish_all/etc. feedbacks are ignored
        if cur_inst is not None:
            pairs.append((cur_inst, cur_mistakes))
    return pairs


def load_unique_target_instructions(plan_set: str, split: str) -> List[str]:
    """Return the sorted unique set of instruction texts in the target split."""
    ds = load_dataset(HF_DATASET_NAME, plan_set, split=split)
    bag: set[str] = set()
    for row in ds:
        for t, ty in zip(row["output_texts"], row["output_types"]):
            if ty == "instruction":
                bag.add(t)
    return sorted(bag)


def embed_texts(
    texts: List[str], embedding_model_id: str, device: str = "cuda:0"
) -> np.ndarray:
    """Compute L2-normalized sentence embeddings via sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "sentence-transformers not installed. Run: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(embedding_model_id, device=device)
    embs = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
    )
    return embs


def retrieve_topk(
    query_emb: np.ndarray, corpus_emb: np.ndarray, top_k: int
) -> np.ndarray:
    """Cosine-similarity top-k indices (embeddings assumed normalized)."""
    sims = query_emb @ corpus_emb.T  # [N_query, N_corpus]
    return np.argsort(-sims, axis=1)[:, :top_k]


def format_examples(retrieved: List[Tuple[str, List[str]]]) -> str:
    """Format retrieved (instruction, mistakes) into the summarizer prompt body."""
    chunks: List[str] = []
    for i, (inst, mistakes) in enumerate(retrieved, start=1):
        chunks.append(f"Example {i}:")
        chunks.append(f"  Instruction: {inst}")
        if mistakes:
            chunks.append("  Annotated mistakes:")
            for m in mistakes:
                chunks.append(f"    - {m}")
        else:
            chunks.append("  Annotated mistakes: (none observed in training data)")
        chunks.append("")
    return "\n".join(chunks).strip()


@torch.no_grad()
def summarize_one(
    instruction: str,
    examples_text: str,
    tokenizer,
    model,
    max_new_tokens: int = 256,
) -> str:
    """Run the summarizer LLM once for a single instruction."""
    user_msg = SUMMARIZER_USER_TEMPLATE.format(
        examples=examples_text, instruction=instruction
    )
    messages = [{"role": "user", "content": user_msg}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_trimmed = gen[0][inputs.input_ids.shape[1]:]
    out = tokenizer.decode(gen_trimmed, skip_special_tokens=True).strip()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build per-instruction mistake-summary cache for the agentic mistake "
            "detector (paper-following pipeline; see file docstring)."
        )
    )
    p.add_argument("--plan_set", default="main",
                   choices=["main", "advanced_planning"],
                   help="HF dataset plan_set.")
    p.add_argument("--split", default="test",
                   choices=["train", "validation", "test"],
                   help="Target split to build summaries for.")
    p.add_argument("--retrieval_source_split", default="train",
                   choices=["train", "validation", "test"],
                   help="Split used as the retrieval corpus. Default 'train' "
                        "(paper convention).")
    p.add_argument("--llm_model_id", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                   help="Text LLM used to summarize mistakes. Paper used "
                        "Qwen-2.5-32B-Instruct; we default to Qwen3 MoE for speed.")
    p.add_argument("--embedding_model_id",
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence embedding model for instruction similarity.")
    p.add_argument("--top_k", type=int, default=5,
                   help="Number of similar instructions to retrieve per target.")
    p.add_argument("--output_path", required=True,
                   help="Where to write the {instruction: summary} JSON cache.")
    p.add_argument("--device", default="cuda:0",
                   help="Device for the embedder + LLM.")
    p.add_argument("--llm_max_new_tokens", type=int, default=256,
                   help="Generation cap per summary call.")
    p.add_argument("--cache_dir", default=None,
                   help="HF cache_dir for downloading the LLM weights.")
    p.add_argument("--resume", action="store_true",
                   help="If set, skip instructions already present in --output_path "
                        "(useful for interrupted runs).")
    p.add_argument("--limit", type=int, default=None,
                   help="(Debug) only summarize the first N unique instructions.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # 1. Build retrieval corpus from the source split
    print(f"[1/5] Loading retrieval source: {args.plan_set} / {args.retrieval_source_split}")
    retrieval_pairs = load_instruction_mistake_pairs(args.plan_set, args.retrieval_source_split)
    print(f"      -> {len(retrieval_pairs)} (instruction, mistakes) pairs")

    # 2. Collect unique target instructions
    print(f"[2/5] Loading target split: {args.plan_set} / {args.split}")
    target_instructions = load_unique_target_instructions(args.plan_set, args.split)
    print(f"      -> {len(target_instructions)} unique target instructions")
    if args.limit is not None:
        target_instructions = target_instructions[: args.limit]
        print(f"      -> truncated to first {len(target_instructions)} (--limit)")

    # 3. Resume cache
    summaries: Dict[str, str] = {}
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path) as f:
            summaries = json.load(f)
        print(f"[resume] loaded {len(summaries)} existing summaries")
    remaining = [i for i in target_instructions if i not in summaries]
    print(f"[2.5] remaining to summarize: {len(remaining)}")

    if not remaining:
        print("Nothing to do. Exiting.")
        return

    # 4. Embeddings + retrieval (only over remaining targets to save time on resume)
    print(f"[3/5] Embedding {len(retrieval_pairs)} retrieval + "
          f"{len(remaining)} target instructions")
    retrieval_texts = [p[0] for p in retrieval_pairs]
    retrieval_emb = embed_texts(retrieval_texts, args.embedding_model_id, args.device)
    target_emb = embed_texts(remaining, args.embedding_model_id, args.device)
    topk_idx = retrieve_topk(target_emb, retrieval_emb, args.top_k)

    # 5. Load summarizer LLM
    print(f"[4/5] Loading summarizer LLM: {args.llm_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_id, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_id,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
    ).eval()

    # 6. Summarize
    print(f"[5/5] Summarizing")
    for i, instruction in enumerate(tqdm(remaining, desc="summaries")):
        retrieved = [retrieval_pairs[j] for j in topk_idx[i]]
        examples_text = format_examples(retrieved)
        out = summarize_one(
            instruction=instruction,
            examples_text=examples_text,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.llm_max_new_tokens,
        )
        summaries[instruction] = out
        # Incremental atomic save so a crash doesn't lose progress
        tmp = args.output_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        os.replace(tmp, args.output_path)

    print(f"\nDone. Wrote {len(summaries)} summaries -> {args.output_path}")


if __name__ == "__main__":
    main()
