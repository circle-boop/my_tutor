import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _write_summary_line(summary_path: Path, line: str) -> None:
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def run_stage(name: str, command: list[str], cwd: Path, summary_path: Path) -> int:
    print(f"\n{'=' * 72}")
    print(f"[阶段开始] {name}")
    print(f"[执行命令] {' '.join(command)}")
    start = time.time()
    _write_summary_line(summary_path, f"[阶段开始] {name} | start={_fmt_time(start)} | cmd={' '.join(command)}")

    result = subprocess.run(command, cwd=str(cwd))

    end = time.time()
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"[阶段完成] {name} (耗时 {elapsed:.1f}s)")
        _write_summary_line(
            summary_path,
            f"[阶段完成] {name} | status=success | returncode=0 | elapsed_sec={elapsed:.1f} | start={_fmt_time(start)} | end={_fmt_time(end)}",
        )
    else:
        print(f"[阶段失败] {name} (退出码 {result.returncode}, 耗时 {elapsed:.1f}s)")
        _write_summary_line(
            summary_path,
            f"[阶段失败] {name} | status=failed | returncode={result.returncode} | elapsed_sec={elapsed:.1f} | start={_fmt_time(start)} | end={_fmt_time(end)}",
        )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="一键执行视频处理全流程")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某个阶段失败后继续执行后续阶段",
    )
    parser.add_argument(
        "--summary-log",
        default=None,
        help="只保存阶段耗时与状态的摘要日志路径；默认写入 thesis_outputs/logs/pipeline_summary_时间.log",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    log_dir = root / "thesis_outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_log:
        summary_path = Path(args.summary_log)
        if not summary_path.is_absolute():
            summary_path = root / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = log_dir / f"pipeline_summary_{stamp}.log"

    stages = [
        ("1/7 字幕转写与纠错", [py, "ai_tutor_backend/1_transcribe.py"]),
        ("2/7 关键帧提取与VLM解析", [py, "ai_tutor_backend/1_5_extract_keyframes_and_vlm.py"]),
        ("3/7 多模态时间线融合", [py, "ai_tutor_backend/1_6_merge_multimodal_timeline.py"]),
        ("4/7 知识点提炼与题库入库", [py, "ai_tutor_backend/2_extract_knowledge.py"]),
        ("5/7 字幕/视觉向量化入库", [py, "ai_tutor_backend/3_build_vector_db.py"]),
        ("6/7 智能抓取并生成网络补充资料", [py, "ai_tutor_backend/5_web_knowledge_collector.py"]),
        ("7/7 补充资料向量化入库", [py, "ai_tutor_backend/4_add_supplementary_docs.py"]),
    ]

    print("开始执行视频处理全流程...")
    print(f"Python 解释器: {py}")
    print(f"项目目录: {root}")
    print(f"摘要日志: {summary_path}")

    run_start = time.time()
    _write_summary_line(summary_path, "视频处理流水线摘要日志")
    _write_summary_line(summary_path, f"run_start={_fmt_time(run_start)}")
    _write_summary_line(summary_path, f"python={py}")
    _write_summary_line(summary_path, f"project_dir={root}")
    _write_summary_line(summary_path, f"continue_on_error={args.continue_on_error}")

    failed = []
    for name, cmd in stages:
        code = run_stage(name, cmd, root, summary_path)
        if code != 0:
            failed.append((name, code))
            if not args.continue_on_error:
                break

    print(f"\n{'=' * 72}")
    run_end = time.time()
    total_elapsed = run_end - run_start
    if failed:
        print("流程执行结束：存在失败阶段")
        for name, code in failed:
            print(f"- {name}: 退出码 {code}")
        _write_summary_line(summary_path, f"run_end={_fmt_time(run_end)}")
        _write_summary_line(summary_path, f"total_elapsed_sec={total_elapsed:.1f}")
        _write_summary_line(summary_path, "run_status=failed")
        for name, code in failed:
            _write_summary_line(summary_path, f"failed_stage={name} | returncode={code}")
        print(f"摘要日志已保存: {summary_path}")
        return 1

    print("流程执行结束：全部阶段成功")
    _write_summary_line(summary_path, f"run_end={_fmt_time(run_end)}")
    _write_summary_line(summary_path, f"total_elapsed_sec={total_elapsed:.1f}")
    _write_summary_line(summary_path, "run_status=success")
    print(f"摘要日志已保存: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
