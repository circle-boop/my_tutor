import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from .config import TRANSCRIPT_DIR, VISUAL_CAPTION_DIR, MM_MERGE_WINDOW_SEC
except ImportError:
    from config import TRANSCRIPT_DIR, VISUAL_CAPTION_DIR, MM_MERGE_WINDOW_SEC


def _nearest_visual(ts: float, visuals: List[Dict[str, Any]], window: float) -> Dict[str, Any] | None:
    best = None
    best_gap = 1e9
    for v in visuals:
        vts = float(v.get("timestamp", 0.0) or 0.0)
        gap = abs(vts - ts)
        if gap <= window and gap < best_gap:
            best = v
            best_gap = gap
    return best


def merge_one(video_name: str):
    transcript_path = TRANSCRIPT_DIR / f"{video_name}.json"
    visual_path = VISUAL_CAPTION_DIR / f"{video_name}.json"
    out_path = TRANSCRIPT_DIR / f"{video_name}.mm.json"

    if not transcript_path.exists():
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        subtitles = json.load(f)

    visuals = []
    if visual_path.exists():
        with open(visual_path, "r", encoding="utf-8") as f:
            visuals = (json.load(f) or {}).get("frames", []) or []

    merged = []
    for s in subtitles:
        st = float(s.get("start", 0.0) or 0.0)
        ed = float(s.get("end", st) or st)
        mid = (st + ed) / 2.0

        v = _nearest_visual(mid, visuals, MM_MERGE_WINDOW_SEC)

        visual_text = (v or {}).get("visual_text", "")
        ocr_text = (v or {}).get("ocr_text", "")
        formulas = (v or {}).get("formula_latex", []) or []
        chart_data = (v or {}).get("chart_data", []) or []

        merged.append(
            {
                "start": st,
                "end": ed,
                "text": (s.get("text") or "").strip(),
                "visual_text": visual_text,
                "ocr_text": ocr_text,
                "formula_latex": formulas,
                "chart_data": chart_data,
                "scene_type": (v or {}).get("scene_type", "unknown"),
                "visual_confidence": float((v or {}).get("confidence", 0.0) or 0.0),
                "has_visual": bool(visual_text or ocr_text or formulas or chart_data),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已生成多模态时间线: {out_path.name}")


def process_all():
    for filename in os.listdir(TRANSCRIPT_DIR):
        if filename.endswith(".json") and not filename.endswith(".mm.json"):
            merge_one(filename[:-5])


if __name__ == "__main__":
    process_all()
