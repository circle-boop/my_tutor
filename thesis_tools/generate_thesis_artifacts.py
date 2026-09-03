from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def try_fix_mojibake(text: str) -> str:
    """Heuristically repair common mojibake cases for Chinese labels."""
    if not isinstance(text, str) or not text:
        return text

    original = text

    def cjk_count(s: str) -> int:
        return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")

    candidates = {original}
    for src, dst in [
        ("latin1", "utf-8"),
        ("cp1252", "utf-8"),
        ("latin1", "gbk"),
        ("cp1252", "gbk"),
    ]:
        try:
            candidates.add(original.encode(src, errors="ignore").decode(dst, errors="ignore"))
        except Exception:
            continue

    best = max(candidates, key=lambda s: (cjk_count(s), -s.count("�"), len(s)))
    return best if cjk_count(best) >= cjk_count(original) else original


def normalize_label(text: str) -> str:
    fixed = try_fix_mojibake(text)
    return fixed.strip()


def scene_label_to_zh(scene: str) -> str:
    mapping = {
        "ppt": "课件",
        "unknown": "未知",
        "mixed": "混合场景",
        "chart": "图表",
        "blackboard": "板书",
        "formula": "公式",
        "code": "代码",
    }
    s = normalize_label(scene)
    return mapping.get(s, s)


def configure_matplotlib_for_chinese(plt):
    # Windows-first fallback fonts for Chinese rendering.
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_video_level_stats(root: Path):
    transcript_dir = root / "transcripts"
    visual_dir = root / "visual_captions"
    keyframe_dir = root / "keyframes"

    per_video: dict[str, dict] = defaultdict(dict)

    # Subtitle and multimodal segment counts.
    for p in transcript_dir.glob("*.json"):
        data = safe_read_json(p)
        if not isinstance(data, list):
            continue

        name = p.stem
        if name.endswith(".mm"):
            video_name = name[:-3]
            per_video[video_name]["mm_segments"] = len(data)
            per_video[video_name]["mm_has_visual"] = sum(1 for x in data if isinstance(x, dict) and x.get("has_visual"))
        else:
            per_video[name]["subtitle_segments"] = len(data)

    # Keyframe counts.
    for idx in keyframe_dir.glob("*/index.json"):
        data = safe_read_json(idx)
        if isinstance(data, list):
            per_video[idx.parent.name]["keyframes"] = len(data)

    # VLM confidence and scene distribution.
    scene_counter = Counter()
    for vf in visual_dir.glob("*.json"):
        data = safe_read_json(vf)
        if not isinstance(data, dict):
            continue
        frames = data.get("frames") or []
        confs = []
        local_scene = Counter()
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            confs.append(float(frame.get("confidence", 0) or 0))
            scene = str(frame.get("scene_type", "unknown"))
            local_scene[scene] += 1
            scene_counter[scene] += 1
        per_video[vf.stem]["vlm_frames"] = len(frames)
        per_video[vf.stem]["vlm_conf_avg"] = round(mean(confs), 4) if confs else 0.0
        per_video[vf.stem]["scene_dist"] = dict(local_scene)

    return per_video, scene_counter


def load_quiz_stats(db_path: Path):
    if not db_path.exists():
        return {
            "quiz_total": 0,
            "knowledge_point_distinct": 0,
            "quiz_by_video": {},
        }

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM quizzes")
    quiz_total = int(cur.fetchone()[0])

    cur.execute("SELECT COUNT(DISTINCT knowledge_point) FROM quizzes")
    knowledge_point_distinct = int(cur.fetchone()[0])

    cur.execute("SELECT video_name, COUNT(*) FROM quizzes GROUP BY video_name ORDER BY COUNT(*) DESC")
    quiz_by_video = {row[0]: int(row[1]) for row in cur.fetchall()}

    conn.close()

    return {
        "quiz_total": quiz_total,
        "knowledge_point_distinct": knowledge_point_distinct,
        "quiz_by_video": quiz_by_video,
    }


def load_agent_round_stats(agent_logs_dir: Path):
    rows = []
    if not agent_logs_dir.exists():
        return rows

    for p in sorted(agent_logs_dir.glob("*.json")):
        data = safe_read_json(p)
        if not isinstance(data, dict):
            continue
        video_name = data.get("video_name", p.stem)
        rounds = data.get("rounds") or []
        for item in rounds:
            if not isinstance(item, dict):
                continue
            stats = item.get("stats") or {}
            rows.append(
                {
                    "video_name": video_name,
                    "round": int(item.get("round", 0) or 0),
                    "total_items": int(stats.get("total_items", 0) or 0),
                    "high_quality_items": int(stats.get("high_quality_items", 0) or 0),
                    "unique_domains": int(stats.get("unique_domains", 0) or 0),
                    "avg_quality": float(stats.get("avg_quality", 0) or 0),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def try_make_plots(out_dir: Path, summary: dict, video_rows: list[dict], scene_counter: Counter, agent_rows: list[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not installed, skip PNG plots. CSV and JSON are still generated.")
        return

    configure_matplotlib_for_chinese(plt)

    # Plot 1: quiz count by video
    quiz_by_video = summary.get("quiz_by_video", {})
    if quiz_by_video:
        items = list(quiz_by_video.items())
        names = [normalize_label(k) for k, _ in items]
        vals = [v for _, v in items]
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(names)), vals)
        plt.xticks(range(len(names)), names, rotation=30, ha="right")
        plt.ylabel("题目数量")
        plt.title("各视频题目数量分布")
        plt.tight_layout()
        plt.savefig(out_dir / "quiz_count_by_video.png", dpi=160)
        plt.close()

    # Plot 2: multimodal visual coverage by video
    if video_rows:
        names = [normalize_label(r["video_name"]) for r in video_rows]
        ratios = [float(r["mm_visual_coverage_ratio"]) for r in video_rows]
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(names)), ratios)
        plt.xticks(range(len(names)), names, rotation=30, ha="right")
        plt.ylabel("覆盖率")
        plt.ylim(0, 1)
        plt.title("各视频多模态视觉覆盖率")
        plt.tight_layout()
        plt.savefig(out_dir / "mm_visual_coverage_by_video.png", dpi=160)
        plt.close()

    # Plot 3: scene distribution
    if scene_counter:
        labels_raw = list(scene_counter.keys())
        labels = [scene_label_to_zh(k) for k in labels_raw]
        vals = [scene_counter[k] for k in labels_raw]
        plt.figure(figsize=(8, 5))
        plt.bar(labels, vals)
        plt.ylabel("帧数")
        plt.title("场景类型分布")
        plt.tight_layout()
        plt.savefig(out_dir / "scene_type_distribution.png", dpi=160)
        plt.close()

    # Plot 4: agent quality trend
    if agent_rows:
        by_video = defaultdict(list)
        for r in agent_rows:
            by_video[r["video_name"]].append(r)

        plt.figure(figsize=(8, 5))
        for v, rows in by_video.items():
            rows = sorted(rows, key=lambda x: x["round"])
            x = [i["round"] for i in rows]
            y = [i["high_quality_items"] for i in rows]
            plt.plot(x, y, marker="o", label=normalize_label(v))
        plt.xlabel("轮次")
        plt.ylabel("高质量来源数")
        plt.title("Agent 检索质量趋势")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "agent_quality_trend.png", dpi=160)
        plt.close()


def main():
    root = Path(__file__).resolve().parent.parent
    out_data = root / "thesis_outputs" / "data"
    out_fig = root / "thesis_outputs" / "figures"

    videos = len(list((root / "videos").glob("*.mp4")))
    transcripts = len([p for p in (root / "transcripts").glob("*.json") if not p.name.endswith(".mm.json")])
    transcripts_mm = len(list((root / "transcripts").glob("*.mm.json")))
    visual_json = len(list((root / "visual_captions").glob("*.json")))
    supp_docs = len(list((root / "supplementary_docs").glob("*.md")))
    agent_logs = len(list((root / "supplementary_docs" / "_agent_logs").glob("*.json"))) if (root / "supplementary_docs" / "_agent_logs").exists() else 0

    per_video, scene_counter = load_video_level_stats(root)
    quiz_stats = load_quiz_stats(root / "tutor_database.db")
    agent_rows = load_agent_round_stats(root / "supplementary_docs" / "_agent_logs")

    subtitle_segments_total = sum(int(v.get("subtitle_segments", 0) or 0) for v in per_video.values())
    mm_segments_total = sum(int(v.get("mm_segments", 0) or 0) for v in per_video.values())
    mm_has_visual_total = sum(int(v.get("mm_has_visual", 0) or 0) for v in per_video.values())
    mm_visual_coverage_ratio = round(mm_has_visual_total / mm_segments_total, 4) if mm_segments_total else 0.0

    summary = {
        "videos": videos,
        "transcripts_json": transcripts,
        "transcripts_mm": transcripts_mm,
        "visual_json": visual_json,
        "supp_docs_md": supp_docs,
        "agent_logs": agent_logs,
        "subtitle_segments_total": subtitle_segments_total,
        "mm_segments_total": mm_segments_total,
        "mm_has_visual_segments": mm_has_visual_total,
        "mm_visual_coverage_ratio": mm_visual_coverage_ratio,
        "quiz_total": quiz_stats["quiz_total"],
        "knowledge_point_distinct": quiz_stats["knowledge_point_distinct"],
        "quiz_by_video": quiz_stats["quiz_by_video"],
        "scene_distribution": dict(scene_counter),
    }

    # Per-video table rows.
    video_rows = []
    for name in sorted(per_video.keys()):
        item = per_video[name]
        mm_seg = int(item.get("mm_segments", 0) or 0)
        mm_vis = int(item.get("mm_has_visual", 0) or 0)
        video_rows.append(
            {
                "video_name": name,
                "subtitle_segments": int(item.get("subtitle_segments", 0) or 0),
                "mm_segments": mm_seg,
                "mm_has_visual": mm_vis,
                "mm_visual_coverage_ratio": round(mm_vis / mm_seg, 4) if mm_seg else 0.0,
                "keyframes": int(item.get("keyframes", 0) or 0),
                "vlm_frames": int(item.get("vlm_frames", 0) or 0),
                "vlm_conf_avg": float(item.get("vlm_conf_avg", 0) or 0),
                "quiz_count": int(quiz_stats["quiz_by_video"].get(name, 0) or 0),
            }
        )

    scene_rows = [{"scene_type": k, "count": v} for k, v in sorted(scene_counter.items(), key=lambda x: x[1], reverse=True)]
    quiz_rows = [{"video_name": k, "quiz_count": v} for k, v in quiz_stats["quiz_by_video"].items()]

    out_data.mkdir(parents=True, exist_ok=True)
    (out_data / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv(
        out_data / "video_level_stats.csv",
        video_rows,
        [
            "video_name",
            "subtitle_segments",
            "mm_segments",
            "mm_has_visual",
            "mm_visual_coverage_ratio",
            "keyframes",
            "vlm_frames",
            "vlm_conf_avg",
            "quiz_count",
        ],
    )
    write_csv(out_data / "scene_distribution.csv", scene_rows, ["scene_type", "count"])
    write_csv(out_data / "quiz_distribution.csv", quiz_rows, ["video_name", "quiz_count"])
    write_csv(
        out_data / "agent_round_stats.csv",
        agent_rows,
        ["video_name", "round", "total_items", "high_quality_items", "unique_domains", "avg_quality"],
    )

    try_make_plots(out_fig, summary, video_rows, scene_counter, agent_rows)

    print("[OK] Thesis artifacts generated.")
    print(f"Summary: {out_data / 'metrics_summary.json'}")
    print(f"Tables : {out_data}")
    print(f"Figures: {out_fig}")


if __name__ == "__main__":
    main()
