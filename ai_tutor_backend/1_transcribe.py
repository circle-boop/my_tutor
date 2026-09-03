import os
import json
from openai import OpenAI
from faster_whisper import WhisperModel

try:
    from .config import (
        VIDEO_DIR,
        TRANSCRIPT_DIR,
        LLM_API_KEY,
        LLM_BASE_URL,
        LLM_MODEL_NAME,
        WHISPER_MODEL_PATH,
    )
except ImportError:
    from config import (
        VIDEO_DIR,
        TRANSCRIPT_DIR,
        LLM_API_KEY,
        LLM_BASE_URL,
        LLM_MODEL_NAME,
        WHISPER_MODEL_PATH,
    )

TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None


def save_transcript(json_path, transcript_data, label="字幕"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=4)
    print(f"[OK] {label}已保存至: {json_path}")


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text.strip()


def _parse_corrected_subtitles(result_text: str):
    cleaned = _strip_code_fence(result_text)
    data = json.loads(cleaned)
    if isinstance(data, dict):
        data = data.get("subtitles", data.get("items", data.get("data")))
    if not isinstance(data, list):
        raise ValueError("纠错结果不是字幕列表")
    return data


def correct_transcript_chunk(chunk_data):
    if client is None:
        return chunk_data

    prompt = f"""请作为专业的字幕纠错助手，对以下语音转录得到的视频字幕进行纠错。
只允许修正 text 字段中的错别字、同音误识别和专业术语错误，不要改动 start、end 字段。
必须返回合法 JSON 对象，不要添加 Markdown 标记或额外解释。

返回格式必须严格如下：
{{
  "subtitles": [
    {{"start": 0.0, "end": 1.0, "text": "纠错后的字幕"}}
  ]
}}

待处理的 JSON 数据：
{json.dumps(chunk_data, ensure_ascii=False)}
"""
    try:
        request_kwargs = {
            "model": LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        try:
            response = client.chat.completions.create(
                **request_kwargs,
                response_format={"type": "json_object"},
            )
        except TypeError:
            response = client.chat.completions.create(**request_kwargs)
        result_text = response.choices[0].message.content.strip()
        parsed = _parse_corrected_subtitles(result_text)
        normalized = []
        for original, corrected in zip(chunk_data, parsed):
            if not isinstance(corrected, dict):
                normalized.append(original)
                continue
            normalized.append(
                {
                    "start": original.get("start"),
                    "end": original.get("end"),
                    "text": (corrected.get("text") or original.get("text") or "").strip(),
                }
            )
        if len(normalized) != len(chunk_data):
            raise ValueError(f"纠错结果条数不匹配: expected={len(chunk_data)}, got={len(normalized)}")
        return normalized
    except Exception as e:
        print(f" -> [WARN] 大模型纠错失败或 JSON 解析异常，将使用原版字幕保底。原因: {e}")
        return chunk_data


def process_videos():
    print("正在加载 Whisper 模型...")
    model = WhisperModel(WHISPER_MODEL_PATH, device="cpu", compute_type="int8")

    for filename in os.listdir(VIDEO_DIR):
        if filename.endswith(".mp4"):
            video_path = VIDEO_DIR / filename
            json_filename = filename.replace(".mp4", ".json")
            json_path = TRANSCRIPT_DIR / json_filename

            if json_path.exists():
                print(f"文件已存在，跳过处理: {json_filename}")
                continue

            print(f"\n{'-' * 50}\n开始转录: {filename}")
            segments, _ = model.transcribe(
                str(video_path),
                beam_size=5,
                language="zh",
                initial_prompt="以下是一段普通话教学视频，请务必使用简体中文生成字幕。",
            )

            raw_transcript_data = []
            for segment in segments:
                chunk = {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text,
                }
                raw_transcript_data.append(chunk)
                print(f"[{chunk['start']}s -> {chunk['end']}s] {chunk['text']}")

            print(f"\n[Whisper 原始转录完成，共 {len(raw_transcript_data)} 句，开始交由 LLM 进行专业词汇纠错清洗...]")
            save_transcript(json_path, raw_transcript_data, label="Whisper 原始字幕")

            CHUNK_SIZE = 15
            corrected_transcript_data = []

            for i in range(0, len(raw_transcript_data), CHUNK_SIZE):
                chunk = raw_transcript_data[i:i + CHUNK_SIZE]
                print(f" -> 正在清洗第 {i + 1} 到 {min(i + CHUNK_SIZE, len(raw_transcript_data))} 句字幕...")
                corrected_chunk = correct_transcript_chunk(chunk)

                if len(corrected_chunk) == len(chunk):
                    corrected_transcript_data.extend(corrected_chunk)
                else:
                    corrected_transcript_data.extend(chunk)

                # Keep a valid transcript file on disk even if a later chunk is interrupted.
                remaining_raw = raw_transcript_data[i + CHUNK_SIZE:]
                save_transcript(json_path, corrected_transcript_data + remaining_raw, label="阶段性字幕")

            save_transcript(json_path, corrected_transcript_data, label="纠错清洗完成，字幕")


if __name__ == "__main__":
    process_videos()
