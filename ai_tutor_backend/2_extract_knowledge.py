import os
import json
import sqlite3
from openai import OpenAI

try:
    from .config import TRANSCRIPT_DIR, DB_PATH, LLM_API_KEY, LLM_BASE_URL, QUIZ_MODEL_NAME
except ImportError:
    from config import TRANSCRIPT_DIR, DB_PATH, LLM_API_KEY, LLM_BASE_URL, QUIZ_MODEL_NAME

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT,
            start_time REAL,
            end_time REAL,
            knowledge_point TEXT,
            question TEXT,
            options TEXT,
            answer TEXT
        )
    '''
    )
    conn.commit()
    return conn


def call_llm_for_quiz(text_block):
    if client is None:
        raise RuntimeError("LLM 未配置。请在 .env 中设置 LLM_API_KEY。")

    prompt = f"""
    你是一个教育专家。请阅读以下视频课程片段（可能包含语音字幕 + 图表/公式视觉信息）：
    {text_block}

    请完成以下任务：
    1. 提取这段内容的核心知识点（限50字以内）。
    2. 基于这段内容中**实际讲解的知识**，出一道单项选择题，用于考察学生是否听懂。
    3. 【出题质量要求】：
       - 题目必须针对视频中实质性讲解的**核心概念或原理**。
       - **绝对禁止**出诸如“视频里有没有提到某人/某物”、“以上皆非”、“无法判断”、“视频未提及”这种无聊或无意义的选项/陷阱题。
       - 如果片段中有图像/图表/公式信息，题目应优先结合这些关键信息，不只复述语音。

    请严格按照以下 JSON 格式输出，不要输出任何额外的解释文字或 Markdown 标记（不要输出 ```json ）：
    {{
        "knowledge_point": "知识点总结",
        "question": "题干内容",
        "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
        "answer": "A"
    }}
    """
    try:
        response = client.chat.completions.create(
            model=QUIZ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        return json.loads(result_text)
    except Exception as e:
        print(f"大模型调用或 JSON 解析失败: {e}")
        return None


def process_and_extract():
    conn = init_db()
    cursor = conn.cursor()

    for filename in os.listdir(TRANSCRIPT_DIR):
        if not filename.endswith(".json") or filename.endswith(".mm.json") or filename.endswith(".v.json"):
            continue

        video_name = filename.replace(".json", "")

        cursor.execute("SELECT COUNT(*) FROM quizzes WHERE video_name = ?", (video_name,))
        if cursor.fetchone()[0] > 0:
            print(f"跳过: 视频 [{video_name}] 的题库已存在。")
            continue

        mm_path = TRANSCRIPT_DIR / f"{video_name}.mm.json"
        file_path = mm_path if mm_path.exists() else (TRANSCRIPT_DIR / filename)
        print(f"\n开始处理新视频字幕: {video_name} | 数据源: {file_path.name}")

        with open(file_path, "r", encoding="utf-8") as f:
            subtitles = json.load(f)

        CHUNK_SIZE = 20
        for i in range(0, len(subtitles), CHUNK_SIZE):
            chunk = subtitles[i:i + CHUNK_SIZE]
            if not chunk:
                continue

            start_time = chunk[0]["start"]
            end_time = chunk[-1]["end"]
            text_lines = []
            for item in chunk:
                t = (item.get("text") or "").strip()
                v = (item.get("visual_text") or "").strip()
                ocr = (item.get("ocr_text") or "").strip()
                formulas = item.get("formula_latex") or []
                charts = item.get("chart_data") or []
                seg = t
                if v:
                    seg += f"\n[视觉] {v}"
                if ocr:
                    seg += f"\n[OCR] {ocr}"
                if formulas:
                    seg += "\n[公式] " + " ; ".join([str(x) for x in formulas if x])
                if charts:
                    seg += "\n[图表] " + " ; ".join([str(x) for x in charts if x])
                text_lines.append(seg)

            text_block = "\n".join([x for x in text_lines if x.strip()])

            print(f"正在提炼 [{start_time}s - {end_time}s] 的知识点...")
            qa_data = call_llm_for_quiz(text_block)

            if qa_data:
                options_str = json.dumps(qa_data["options"], ensure_ascii=False)
                cursor.execute(
                    '''
                    INSERT INTO quizzes (video_name, start_time, end_time, knowledge_point, question, options, answer)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                    (
                        video_name,
                        start_time,
                        end_time,
                        qa_data["knowledge_point"],
                        qa_data["question"],
                        options_str,
                        qa_data["answer"],
                    ),
                )
                conn.commit()
                print(" -> 入库成功")

    conn.close()
    print("\n所有视频知识提炼完毕，已存入 tutor_database.db")


if __name__ == "__main__":
    process_and_extract()
