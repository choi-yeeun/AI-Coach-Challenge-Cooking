"""
grounding_dino_verifier.py — Post-hoc visual grounding verification for VLM mistake claims.

Wraps HuggingFace's GroundingDinoForObjectDetection with two entry points:
- ``verify(frame_path, objects) -> bool``: just the keep/drop decision (back-compat).
- ``verify_detailed(frame_path, objects, save_viz_path=None) -> dict``: per-query
  detections + optional bbox-overlay visualization saved to disk.

Conservative rule: empty/missing object list → True (keep). At least one of the
listed objects detected above the score threshold → True (keep). No listed
object detected at all → False (drop).

Strategy: each object is queried INDIVIDUALLY (not concatenated). Empirical
test on a featureless frame showed concatenated queries 'a. b. c.' produce
spurious matches on the last phrase, whereas single-phrase queries 'a.' alone
are clean. The trade-off is N forward passes per verify (N = #objects),
typically 2-4, which is sparse since verify is only called per Feedback emit.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, GroundingDinoForObjectDetection


DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


class GroundingDinoVerifier:
    """Lazy-loaded DINO verifier. One instance per inference worker process."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        box_threshold: float = 0.4,
        text_threshold: float = 0.25,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = (
            GroundingDinoForObjectDetection.from_pretrained(
                model_id, cache_dir=cache_dir, torch_dtype=torch.float32
            )
            .to(device)
            .eval()
        )
        print(f"[grounding_dino] loaded {model_id} on {device}; box_thr={box_threshold}")

    def _post_process(self, outputs, inputs, image):
        """Call DINO post-processing with kwarg compat (threshold vs box_threshold)."""
        try:
            return self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],
            )
        except TypeError:
            return self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],
            )

    @torch.no_grad()
    def verify_detailed(
        self,
        frame_path: str,
        expected_objects: List[str],
        save_viz_path: Optional[str] = None,
        feedback_text: str = "",
    ) -> Dict[str, Any]:
        """Run DINO per-query and return decision + per-query detection info.

        Return shape::

            {
              "decision": bool,                   # True = keep, False = drop
              "frame_path": str,
              "per_query": [
                  {"query": "spoon.",
                   "n_dets": 0,
                   "scores": [],
                   "labels": [],
                   "boxes": []},                  # list of [x1,y1,x2,y2]
                  ...
              ],
              "feedback_text": str,
            }

        If ``save_viz_path`` is given, a bbox-overlay JPG is written to that
        path AFTER the decision is computed (header includes KEEP/DROP).
        """
        result: Dict[str, Any] = {
            "decision": True,
            "frame_path": frame_path,
            "per_query": [],
            "feedback_text": feedback_text,
        }

        if not expected_objects:
            # nothing to verify -> keep, no viz
            return result

        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception as e:
            print(f"[grounding_dino] failed to open {frame_path}: {e}; keeping emit")
            return result

        decision = False
        for obj in expected_objects:
            phrase = obj.strip().lower()
            if not phrase:
                continue
            query = phrase if phrase.endswith(".") else phrase + "."

            inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            results = self._post_process(outputs, inputs, image)

            scores = results[0].get("scores", torch.tensor([]))
            # HF >= 4.51 prefers 'text_labels'; older 'labels' may be int ids
            labels = results[0].get("text_labels", None)
            if labels is None:
                labels = results[0].get("labels", [])
            boxes = results[0].get("boxes", torch.tensor([]))

            scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
            labels_list = [str(x) for x in (labels if labels is not None else [])]
            boxes_list = (
                boxes.tolist() if hasattr(boxes, "tolist") else [list(b) for b in boxes]
            )

            result["per_query"].append({
                "query": query,
                "n_dets": len(scores_list),
                "scores": scores_list,
                "labels": labels_list,
                "boxes": boxes_list,
            })
            if len(scores_list) > 0:
                decision = True
                # keep going through remaining objects so the visualization
                # shows all evidence — small extra cost (N typically 2-4)

        result["decision"] = decision

        if save_viz_path is not None:
            try:
                draw_dino_detections(image, result, save_viz_path)
            except Exception as e:
                print(f"[grounding_dino] viz save failed for {save_viz_path}: {e}")

        return result

    def verify(self, frame_path: str, expected_objects: List[str]) -> bool:
        """Bool-only wrapper for back-compat."""
        return self.verify_detailed(frame_path, expected_objects)["decision"]


# ---- visualization helper ----

_QUERY_COLORS = [
    (255, 50, 50),    # red
    (50, 255, 50),    # green
    (50, 150, 255),   # blue
    (255, 165, 0),    # orange
    (200, 50, 200),   # purple
    (50, 200, 200),   # teal
]


def draw_dino_detections(image: Image.Image, result: Dict[str, Any], save_path: str) -> None:
    """Draw bboxes + score/label per query, save as JPG. Header shows KEEP/DROP + feedback."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for i, q in enumerate(result["per_query"]):
        color = _QUERY_COLORS[i % len(_QUERY_COLORS)]
        for score, label, box in zip(q["scores"], q["labels"], q["boxes"]):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            tag = f"{label} {score:.2f}"
            # text background for readability
            tb = draw.textbbox((x1, max(y1 - 16, 0)), tag, font=font_small)
            draw.rectangle(tb, fill=color)
            draw.text((x1, max(y1 - 16, 0)), tag, fill="white", font=font_small)

    # Top-left header strip
    decision_tag = "KEEP" if result["decision"] else "DROP"
    queries_str = ", ".join(q["query"].rstrip(".") for q in result["per_query"])
    header_line1 = f"[{decision_tag}] queries: {queries_str}"
    fb = result.get("feedback_text", "")
    header_line2 = f"feedback: {fb[:100]}"
    header_height = 38
    draw.rectangle([0, 0, img.width, header_height], fill=(0, 0, 0))
    header_color = (50, 255, 50) if result["decision"] else (255, 80, 80)
    draw.text((5, 2), header_line1, fill=header_color, font=font)
    draw.text((5, 20), header_line2, fill="white", font=font_small)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    img.save(save_path, "JPEG", quality=85)
