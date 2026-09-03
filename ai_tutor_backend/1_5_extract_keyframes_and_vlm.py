import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed  # 新增：用于多线程并发

import cv2
import numpy as np
from openai import OpenAI

try:
    from .config import (
        VIDEO_DIR,
        KEYFRAME_DIR,
        VISUAL_CAPTION_DIR,
        LLM_API_KEY,
        LLM_BASE_URL,
        VLM_MODEL_NAME,
        ENABLE_VLM,
        KEYFRAME_MIN_INTERVAL_SEC,
        KEYFRAME_MAX_PER_VIDEO,
    )
except ImportError:
    from config import (
        VIDEO_DIR,
        KEYFRAME_DIR,
        VISUAL_CAPTION_DIR,
        LLM_API_KEY,
        LLM_BASE_URL,
        VLM_MODEL_NAME,
        ENABLE_VLM,
        KEYFRAME_MIN_INTERVAL_SEC,
        KEYFRAME_MAX_PER_VIDEO,
    )

KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_CAPTION_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if (LLM_API_KEY and ENABLE_VLM) else None

# ----------------- 优化 1：图像物理压缩 -----------------
def _encode_image(path: Path) -> str:
    """读取图片并限制最大尺寸，转为 Base64 以降低 API 传输耗时和 Token 开销"""
    # 修复：cv2.imread 无法读取中文路径，改用 numpy + imdecode 读取
    img_array = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return ""

    # 限制最长边为 1024 像素
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 压缩为 JPEG（质量设为 80），直接在内存中转为 Base64
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode("utf-8")


def _compute_dhash(gray_frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray_frame, (9, 8))
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def _compute_edge_density(gray_frame: np.ndarray) -> float:
    edges = cv2.Canny(gray_frame, 50, 150)
    density = np.count_nonzero(edges) / edges.size
    return density


def _extract_keyframes(video_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0

    keyframes: List[Dict[str, Any]] = []
    last_anchor_hash = None
    last_saved_ts = -999.0

    # 调优：大幅度提高校验阈值，屏蔽无效讲师动作和低密度无字画面
    HASH_DIFF_THRESHOLD = 15   
    MIN_EDGE_DENSITY = 0.025   

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = idx / fps
        idx += 1

        if len(keyframes) >= KEYFRAME_MAX_PER_VIDEO:
            break

        if (ts - last_saved_ts) < KEYFRAME_MIN_INTERVAL_SEC:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_hash = _compute_dhash(gray)
        
        should_save = False
        reason = ""

        if last_anchor_hash is None:
            should_save = True
            reason = "first_frame"
        else:
            edge_density = _compute_edge_density(gray)
            if edge_density < MIN_EDGE_DENSITY:
                continue 

            hamming_distance = np.count_nonzero(current_hash != last_anchor_hash)
            
            if hamming_distance >= HASH_DIFF_THRESHOLD:
                should_save = True
                reason = f"hash_diff={hamming_distance}, edge_den={edge_density:.3f}"

        if should_save:
            image_name = f"{int(ts * 1000):010d}.jpg"
            image_path = out_dir / image_name
            cv2.imencode('.jpg', frame)[1].tofile(str(image_path))
            
            keyframes.append(
                {
                    "timestamp": round(ts, 2),
                    "frame_path": str(image_path),
                    "reason": reason,
                }
            )
            last_anchor_hash = current_hash
            last_saved_ts = ts

    cap.release()

    if duration > 0 and len(keyframes) == 0:
        print(f"[WARN] 未检测到有效结构变化，视频时长={duration:.1f}s，可能属于纯口述/实拍视频")

    return keyframes


def _vlm_describe_frame(image_path: Path) -> Dict[str, Any]:
    if client is None:
        return {"scene_type": "unknown", "visual_text": "", "ocr_text": "", "formula_latex": [], "chart_data": [], "confidence": 0.0}

    b64 = _encode_image(image_path)
    prompt = (
        "你是教学视频视觉解析助手。请从画面中提取对学习有用的信息，"
        "特别是图表、坐标轴、函数图像、公式、板书结论。"
        "仅输出JSON，不要markdown，不要解释。"
        "JSON结构固定："
        "{\"scene_type\":\"ppt|blackboard|chart|formula|code|mixed|unknown\","
        "\"visual_text\":\"...\",\"ocr_text\":\"...\","
        "\"formula_latex\":[\"...\"],\"chart_data\":[\"...\"],\"confidence\":0.0}"
    )

    try:
        resp = client.chat.completions.create(
            model=VLM_MODEL_NAME,
            temperature=0.1,
            max_tokens=800, # 从300提高到800：确保信息量密集的画面有足够的Token输出完整JSON结构
            timeout=60.0, # [新增] 强制60秒超时机制，彻底解决因网络波动导致的线程死锁“卡住”问题
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        if txt.startswith("```json"):
            txt = txt[7:-3].strip()
        elif txt.startswith("```"):
            txt = txt[3:-3].strip()

        # [新增] 暴力清理 VLM 有时会在末尾加上的无关解释性文本，确保是个纯净的 JSON 边界
        if txt.find("{") != -1 and txt.rfind("}") != -1:
            txt = txt[txt.find("{") : txt.rfind("}") + 1]

        data = json.loads(txt)
        return {
            "scene_type": data.get("scene_type", "unknown"),
            "visual_text": data.get("visual_text", "") or "",
            "ocr_text": data.get("ocr_text", "") or "",
            "formula_latex": data.get("formula_latex", []) or [],
            "chart_data": data.get("chart_data", []) or [],
            "confidence": float(data.get("confidence", 0.0) or 0.0),
        }
    except Exception as e:
        print(f"[WARN] VLM 解析失败 {image_path.name}: {e}")
        return {"scene_type": "unknown", "visual_text": "", "ocr_text": "", "formula_latex": [], "chart_data": [], "confidence": 0.0}


# ----------------- 内部工作函数（用于多线程） -----------------
def _process_single_frame(kf: Dict[str, Any]) -> Dict[str, Any]:
    """独立处理单张关键帧，供线程池调用"""
    p = Path(kf["frame_path"])
    vlm = _vlm_describe_frame(p)
    return {
        "timestamp": kf["timestamp"],
        "frame_path": kf["frame_path"],
        "reason": kf.get("reason", ""),
        **vlm,
    }


# ----------------- 优化 2：多线程并发请求 -----------------
def process_videos():
    for filename in os.listdir(VIDEO_DIR):
        if not filename.endswith(".mp4"):
            continue

        video_name = filename[:-4]
        video_path = VIDEO_DIR / filename

        out_kf_dir = KEYFRAME_DIR / video_name
        out_kf_dir.mkdir(parents=True, exist_ok=True)

        visual_json_path = VISUAL_CAPTION_DIR / f"{video_name}.json"
        if visual_json_path.exists():
            print(f"跳过: 视觉结果已存在 {visual_json_path.name}")
            continue

        print(f"\n{'-' * 50}\n[关键帧/VLM] 处理视频: {video_name}")

        keyframes = _extract_keyframes(video_path, out_kf_dir)
        total_frames = len(keyframes)
        print(f"提取关键帧: {total_frames}")

        if total_frames == 0:
            continue

        with open(out_kf_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(keyframes, f, ensure_ascii=False, indent=2)

        frames_out = []
        completed_count = 0
        
        # 并发线程数。建议设置为 3-5。如果 API 报错 429 (Too Many Requests)，请调低此数值。
        MAX_WORKERS = 8 

        print(f"  启动并发解析 (线程数: {MAX_WORKERS})...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务到线程池
            future_to_kf = {executor.submit(_process_single_frame, kf): kf for kf in keyframes}
            
            # as_completed 会在任意线程完成时立刻产出结果（无序输出）
            for future in as_completed(future_to_kf):
                try:
                    result = future.result()
                    frames_out.append(result)
                except Exception as exc:
                    print(f"[WARN] 单帧处理引发异常: {exc}")
                
                completed_count += 1
                
                # 因并发输出是无序的，写入 JSON 前按时间戳强制排序，保证 RAG 时序一致性
                frames_out.sort(key=lambda x: x["timestamp"])

                # 实时增量存档
                checkpoint_data = {
                    "video_name": video_name,
                    "frame_count": len(frames_out),
                    "model": VLM_MODEL_NAME,
                    "status": f"processing {completed_count}/{total_frames}",
                    "frames": frames_out,
                }
                with open(visual_json_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

                if completed_count % 5 == 0 or completed_count == total_frames:
                    print(f"  已解析 {completed_count}/{total_frames} 帧")

        # 闭环状态更新
        final_data = {
            "video_name": video_name,
            "frame_count": len(frames_out),
            "model": VLM_MODEL_NAME,
            "status": "completed",
            "frames": frames_out,
        }
        with open(visual_json_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        print(f"[OK] 视觉结果解析完成并保存: {visual_json_path}")


if __name__ == "__main__":
    process_videos()
