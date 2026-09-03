import os
import re
import sys
import json
import uuid
import time
import shutil
import sqlite3
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from .config import (
        DB_PATH,
        CHROMA_DB_DIR,
        TRANSCRIPT_DIR,
        VIDEO_DIR,
        EMBEDDING_MODEL_PATH,
        LLM_API_KEY,
        LLM_BASE_URL,
        CHAT_MODEL_NAME,
        EVALUATE_MODEL_NAME,
        KEYFRAME_DIR,
        VISUAL_CAPTION_DIR,
        CORS_ALLOW_ORIGINS,
    )
except ImportError:
    from config import (
        DB_PATH,
        CHROMA_DB_DIR,
        TRANSCRIPT_DIR,
        VIDEO_DIR,
        EMBEDDING_MODEL_PATH,
        LLM_API_KEY,
        LLM_BASE_URL,
        CHAT_MODEL_NAME,
        EVALUATE_MODEL_NAME,
        KEYFRAME_DIR,
        VISUAL_CAPTION_DIR,
        CORS_ALLOW_ORIGINS,
    )

# 初始化 FastAPI
app = FastAPI(title="引导式智能助教系统 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if VIDEO_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")


@app.on_event("startup")
def _startup_watch_video_changes():
    t = threading.Thread(target=_video_watch_loop, daemon=True)
    t.start()

# 模型配置
llm_client = None
if LLM_API_KEY:
    llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

vectorstore = None
vectorstore_error = None
try:
    print("正在加载向量数据库...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
    vectorstore = Chroma(persist_directory=str(CHROMA_DB_DIR), embedding_function=embeddings)
    print("向量数据库加载完成。")
except Exception as e:
    vectorstore_error = str(e)
    print(f"[WARN] 向量数据库加载失败：{vectorstore_error}")


# --- 数据模型定义 ---
class UserAnswerReq(BaseModel):
    question: str
    options: List[str]
    standard_answer: str
    user_answer: str


class Message(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    query: str
    video_name: Optional[str] = None
    history: List[Message] = []
    current_time: Optional[float] = None


# --- 辅助函数：格式化时间 ---
def format_time(seconds):
    try:
        sec = int(float(seconds))
        m, s = divmod(sec, 60)
        return f"{m}min{s}s" if m > 0 else f"{s}s"
    except Exception:
        return f"{seconds}s"


def _is_summary_intent(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    summary_markers = [
        "总结",
        "概括",
        "梳理",
        "归纳",
        "提炼",
        "复盘",
        "大纲",
        "要点",
        "重点",
        "知识点",
        "核心内容",
        "主要内容",
    ]
    return any(k in t for k in summary_markers)


def _is_brief_ack(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if len(t) > 12:
        return False
    ack_words = {
        "嗯", "恩", "是", "是的", "对", "对的", "有", "有过", "好", "好的", "行", "可以", "知道了", "明白", "明白了", "收到"
    }
    return t in ack_words


def _is_entity_like_query(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if "?" in t or "？" in t:
        return False
    if len(t) > 14:
        return False
    has_ascii = any(("a" <= ch.lower() <= "z") for ch in t)
    keywords = ["AI", "ai", "地图", "导航", "豆包", "抖音", "快手", "微信", "ChatGPT", "通义", "文心", "Kimi", "Siri", "小爱"]
    hit_kw = any(k in t for k in keywords)
    return has_ascii or hit_kw


def _sample_transcript_windows(subtitles: List[Dict[str, Any]], centers: List[float], radius: float = 25.0) -> str:
    if not subtitles:
        return ""

    blocks = []
    for c in centers:
        start_window = max(0.0, float(c) - radius)
        end_window = float(c) + radius
        window_texts = []
        for item in subtitles:
            if item.get("start", 0) <= end_window and item.get("end", 0) >= start_window:
                tx = (item.get("text") or "").strip()
                if tx:
                    window_texts.append(tx)
        if window_texts:
            blocks.append(
                f"- 片段（{format_time(start_window)}-{format_time(end_window)}）：" + " ".join(window_texts)
            )

    return "\n".join(blocks)


def _fetch_video_knowledge_points(video_name: str, limit: int = 12) -> List[str]:
    if not video_name:
        return []

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT knowledge_point FROM quizzes WHERE video_name = ? ORDER BY start_time ASC LIMIT ?",
            (video_name, int(limit)),
        )
        rows = cursor.fetchall()
        conn.close()

        seen = set()
        out = []
        for (kp,) in rows:
            kp = (kp or "").strip()
            if not kp:
                continue
            if kp in seen:
                continue
            seen.add(kp)
            out.append(kp)
        return out
    except Exception:
        return []


def _safe_similarity_search(query: str, k: int = 3, filter_dict: Optional[Dict[str, Any]] = None):
    if vectorstore is None:
        return []
    try:
        if filter_dict:
            return vectorstore.similarity_search(query=query, k=k, filter=filter_dict)
        return vectorstore.similarity_search(query=query, k=k)
    except Exception:
        if filter_dict:
            try:
                return vectorstore.similarity_search(query=query, k=max(3, min(k, 8)))
            except Exception:
                return []
        return []


def _extract_option_text(options: List[str], answer_letter: str) -> str:
    target = (answer_letter or "").strip().upper()
    if not target:
        return ""
    for opt in options or []:
        s = (opt or "").strip()
        if not s:
            continue
        if s.upper().startswith(target + "."):
            return s[len(target) + 1 :].strip()
        if s.upper().startswith(target + "、"):
            return s[len(target) + 1 :].strip()
    return ""


# --- 接口 1：动态获取本地视频列表 ---
@app.get("/api/videos")
def get_video_list():
    if not VIDEO_DIR.exists():
        return {"videos": []}
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    video_names = [f[:-4] for f in video_files]
    video_names.sort()
    return {"videos": video_names}


# --- 接口 2：获取视频元数据与提问时间戳 ---
@app.get("/api/video/{video_name}/metadata")
def get_video_metadata(video_name: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT end_time FROM quizzes WHERE video_name = ? ORDER BY end_time ASC", (video_name,))
    rows = cursor.fetchall()
    conn.close()
    return {"video_name": video_name, "timestamps": [row[0] for row in rows] if rows else []}


# --- 接口 3：根据时间点获取具体题目 ---
@app.get("/api/video/{video_name}/question")
def get_question(video_name: str, time: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT knowledge_point, question, options, answer
        FROM quizzes
        WHERE video_name = ? AND abs(end_time - ?) < 1.0
        LIMIT 1
    """,
        (video_name, time),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="该时间点无题目")

    import json

    return {
        "knowledge_point": row[0],
        "question": row[1],
        "options": json.loads(row[2]),
        "standard_answer": row[3],
    }


# --- 接口 4：评估学生答题情况 ---
@app.post("/api/evaluate")
def evaluate_answer(req: UserAnswerReq):
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM 未配置。请在 .env 中设置 LLM_API_KEY。")

    # 先用代码对选择题进行准确判定，防止大模型自由发挥误判
    user_ans = req.user_answer.strip().upper()
    std_ans = req.standard_answer.strip().upper()
    
    is_programmatic_correct = False
    if user_ans.startswith(std_ans + ".") or user_ans.startswith(std_ans + " ") or user_ans == std_ans or std_ans in user_ans:
        is_programmatic_correct = True

    correct_directive = f"已知学生的回答是 **{'正确' if is_programmatic_correct else '错误'}** 的。"

    prompt = f"""
    题目：{req.question}
    所有的选项：{', '.join(req.options)}
    标准答案：{req.standard_answer}
    学生的回答：{req.user_answer}

    {correct_directive}
    请根据判定结果，生成针对知识点的分析反馈（feedback）。

    【强烈要求】：
    1. 如果回答正确：给予肯定，并简述该决定背后的知识原理。
    2. 如果回答错误：必须以“正确答案是：{req.standard_answer}。 ”开头，**并紧接着用自己的话解释为什么该选项才是对的，给出知识点解析**（千万不要仅仅把对应的选项文字重复一遍！必须带有原理解释！）。
    3. 整体字数控制在 50~80 字左右，语气以鼓励为主。

    严格按以下JSON格式输出，不要包含Markdown：
    {{"is_correct": {"true" if is_programmatic_correct else "false"}, "feedback": "反馈内容"}}
    """
    try:
        response = llm_client.chat.completions.create(
            model=EVALUATE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        import json

        data = json.loads(result_text)
        feedback = (data.get("feedback") or "").strip()

        if is_programmatic_correct:
            if not feedback:
                feedback = "回答正确。你抓住了核心知识点，继续保持这种分析方式。"
        else:
            std = std_ans
            correct_option_text = _extract_option_text(req.options, std)
            wrong_prefix = f"正确答案是：{std}。"
            # 无论模型是否遗忘，都强制给出正确答案与解释
            if not feedback:
                if correct_option_text:
                    feedback = (
                        f"{wrong_prefix} 该选项对应的核心概念是“{correct_option_text}”，"
                        "需要结合题干中的知识点来判断其成立条件与适用场景。"
                    )
                else:
                    feedback = (
                        f"{wrong_prefix} 请回到题干中的核心概念，"
                        "从定义、关键特征和适用条件三个方面重新判断。"
                    )
            elif not feedback.startswith("正确答案是："):
                feedback = f"{wrong_prefix} {feedback}"

            # 避免“只给答案不解释”
            core_feedback = feedback.replace(wrong_prefix, "").strip()
            if len(core_feedback) < 12:
                if correct_option_text:
                    feedback = (
                        f"{wrong_prefix} 该题考查的是“{correct_option_text}”这一本质含义，"
                        "错误选项通常混淆了概念边界或忽略了成立前提。"
                    )
                else:
                    feedback = (
                        f"{wrong_prefix} 这道题的关键在于识别概念边界，"
                        "不要只看字面相似，需要对照定义和条件进行判断。"
                    )

        return {
            "is_correct": bool(is_programmatic_correct),
            "feedback": feedback,
        }
    except Exception as e:
        # 兜底：即使 LLM 失败，也要保证错误分支给出正确答案与简要解释
        if is_programmatic_correct:
            return {
                "is_correct": True,
                "feedback": "回答正确。思路是对的，继续保持。",
            }
        std = std_ans
        correct_option_text = _extract_option_text(req.options, std)
        if correct_option_text:
            fallback_feedback = (
                f"正确答案是：{std}。该题考查“{correct_option_text}”的核心含义，"
                "建议你对照题干中的关键条件再次核对。"
            )
        else:
            fallback_feedback = (
                f"正确答案是：{std}。该题需要先明确概念定义，再根据题干条件逐项排除。"
            )
        return {
            "is_correct": False,
            "feedback": fallback_feedback,
        }


# --- 接口 5：结合 RAG 向量数据库的问答 ---
@app.post("/api/chat")
def chat_with_rag(req: ChatReq):
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM 未配置。请在 .env 中设置 LLM_API_KEY。")
    if vectorstore is None:
        raise HTTPException(status_code=503, detail=f"向量数据库不可用: {vectorstore_error}")

    if _is_brief_ack(req.query):
        return {"answer": "收到。继续聚焦课程内容，可以直接问概念区别、原理、例题或让我做要点总结。"}

    user_query = req.query
    if _is_entity_like_query(req.query):
        user_query = f"请基于当前课程标准判断：\"{req.query.strip()}\"是否属于真正的人工智能系统。先给明确结论（是/不是/不完全是），再给2-3条简要理由。"

    is_summary = _is_summary_intent(user_query)

    # =============== RAG 混合路检索策略 ===============
    all_retrieved_docs = []
    
    if is_summary and req.video_name:
        # 【路线A：总结类意图】优先找当前视频，保证不串台
        docs_video = _safe_similarity_search(
            query=f"{req.video_name} {user_query}",
            k=12,
            filter_dict={"source_type": "video_multimodal", "video_name": req.video_name},
        )
        if not docs_video:
            docs_video = _safe_similarity_search(
                query=f"{req.video_name} {user_query}",
                k=12,
                filter_dict={"source_type": "video_subtitle", "video_name": req.video_name},
            )
        all_retrieved_docs.extend(docs_video)
        
        # 即使是总结，也去补充资料和外部看一眼有没有大纲定义 (强塞 2 条)
        docs_supp = _safe_similarity_search(query=user_query, k=2, filter_dict={"source_type": "supplementary_doc"})
        all_retrieved_docs.extend(docs_supp)

    else:
        # 【路线B：常规提问】分路检索，强制掺入“当前视频”、“补充资料”和“全库其他”
        
        # 1. 搜当前视频片段 (优先多模态，提取Top 3)
        if req.video_name:
            docs_curr = _safe_similarity_search(
                query=user_query, 
                k=3, 
                filter_dict={"source_type": "video_multimodal", "video_name": req.video_name}
            )
            if not docs_curr:
                docs_curr = _safe_similarity_search(
                    query=user_query, 
                    k=3, 
                    filter_dict={"source_type": "video_subtitle", "video_name": req.video_name}
                )
            all_retrieved_docs.extend(docs_curr)
            
        # 2. 搜补充资料 (强制掺入教辅或定义，提取Top 2)
        docs_supp = _safe_similarity_search(query=user_query, k=2, filter_dict={"source_type": "supplementary_doc"})
        all_retrieved_docs.extend(docs_supp)
        
        # 3. 如果总线少于3条（可能刚开始看没资料），做一次全库保底打捞
        if len(all_retrieved_docs) < 3:
            docs_global = _safe_similarity_search(query=user_query, k=3)
            # 为了防止重复，简单按page_content去重
            seen_content = {d.page_content for d in all_retrieved_docs}
            for d in docs_global:
                if d.page_content not in seen_content:
                    all_retrieved_docs.append(d)
                    seen_content.add(d.page_content)
    
    docs = all_retrieved_docs
    # ==================================================

    if not docs:
        context_str = "未检索到相关资料。"
    else:
        context_list = []
        for doc in docs:
            source_type = doc.metadata.get("source_type", "video_subtitle")
            if source_type in ("video_subtitle", "video_multimodal"):
                v_name = doc.metadata.get('video_name', '未知视频')
                start_str = format_time(doc.metadata.get('start_time', 0))
                end_str = format_time(doc.metadata.get('end_time', 0))
                tag = "多模态" if source_type == "video_multimodal" else "字幕"
                source_label = f"[{v_name}，时间: {start_str}-{end_str}，{tag}]"
            else:
                f_name = doc.metadata.get('file_name', '补充资料')
                source_label = f"[{f_name}]"

            context_list.append(f"{source_label} {doc.page_content}")

        context_str = "\n".join(context_list)

    local_context = ""
    sampled_context = ""
    quiz_kps = []

    transcript_path = None
    if req.video_name:
        mm_path = TRANSCRIPT_DIR / f"{req.video_name}.mm.json"
        transcript_path = mm_path if mm_path.exists() else (TRANSCRIPT_DIR / f"{req.video_name}.json")

    # 1) 常规：当前时间点附近上下文（±30s）并融合视觉字段
    if req.video_name and req.current_time is not None and transcript_path and transcript_path.exists():
        import json
        with open(transcript_path, "r", encoding="utf-8") as f:
            subtitles = json.load(f)

        start_window = req.current_time - 30
        end_window = req.current_time + 30
        window_texts = []

        for item in subtitles:
            if item.get("start", 0) <= end_window and item.get("end", 0) >= start_window:
                tx = (item.get("text") or "").strip()
                v_text = (item.get("visual_text") or "").strip()
                ocr = (item.get("ocr_text") or "").strip()
                formulas = item.get("formula_latex") or []
                charts = item.get("chart_data") or []
                
                # 融合字幕与同一时间戳的画面内容
                combined = tx
                if v_text:
                    combined += f" [画面分析: {v_text}]"
                if ocr:
                    combined += f" [OCR文字: {ocr}]"
                if formulas:
                    combined += f" [画面公式: {'; '.join(formulas)}]"
                if charts:
                    combined += f" [图表数据: {'; '.join(charts)}]"
                
                if combined.strip():
                    window_texts.append(combined.strip())

        if window_texts:
            local_context = (
                f"\n【当前视频画面上下文（{format_time(start_window)}-{format_time(end_window)}）】：\n"
                + " ".join(window_texts)
            )

    # 2) 总结类：全片采样 + 题库知识点
    if is_summary and req.video_name:
        quiz_kps = _fetch_video_knowledge_points(req.video_name, limit=12)
        if transcript_path and transcript_path.exists():
            import json
            with open(transcript_path, "r", encoding="utf-8") as f:
                subtitles = json.load(f)
            duration = float(subtitles[-1].get("end", 0) or 0) if subtitles else 0.0
            centers = [30.0, duration / 2.0, max(0.0, duration - 30.0)] if duration > 0 else [30.0]
            sampled_context = _sample_transcript_windows(subtitles, centers=centers, radius=22.0)

    quiz_kps_str = "\n".join([f"- {kp}" for kp in (quiz_kps or [])])
    quiz_kps_block = f"\n\n【题库知识点（自动提炼，供总结参考）】\n{quiz_kps_str}" if quiz_kps_str else ""
    sampled_block = f"\n\n【全片采样（用于总结覆盖核心概念）】\n{sampled_context}" if sampled_context else ""

    # 重构版 System Prompt
    system_prompt = f"""
你是一个专业、客观且具备引导能力的视频课程智能助教。请结合【检索资料】、【当前视频画面上下文】与对话历史，为学生提供精准解答。

【回答核心原则】
1. 事实依附：必须严格基于提供的资料与上下文回答。如果资料中未提及，请直接说明“视频中未涉及该部分内容”，禁止自行编造。
2. 结构化呈现：
   - 总结/梳理类问题：请多用 Markdown 列表（- 或 1. 2.）和加粗（**）提取核心概念，确保层级清晰、便于扫读。
   - 概念解释类问题：先给出清晰的定义或结论，再结合视频中的具体案例（如高德地图、自动驾驶等）进行解释。
3. 语气与排版：
   - 保持客观、平等的交流态度。
   - 避免长篇大论，超过 3 行的文字请务必分段。

【意图分流处理】
- 若用户输入为短实体词（如“豆包AI”、“ChatGPT”）：将其理解为“判断该对象是否属于真正AI”的请求。直接给出结论（是/不是/部分是），并分点列出 2-3 条判断依据。
- 若用户输入为短确认（如“嗯”、“懂了”）：简单确认即可，不要展开长篇大论或机械反问。

【上下文信息】
检索资料：
{context_str}

当前视频画面上下文：{local_context}
全片采样参考：{sampled_block}
题库知识点：{quiz_kps_block}
"""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in req.history[-4:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": user_query})

    try:
        response = llm_client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=messages,
            temperature=0.2,
        )
        return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 问答失败: {str(e)}")


# =========================
# 学习请求（小管家 + 搜查官）
# =========================

_PIPELINE_LOCK = threading.Lock()
_LAST_VIDEO_COUNT = 0
_AUTO_PIPELINE_RUNNING = False


class PipelineRunReq(BaseModel):
    continue_on_error: bool = False


class RebuildVideoReq(BaseModel):
    video_name: str
    delete_quiz: bool = True
    delete_vector: bool = True
    delete_generated_files: bool = True


def _run_pipeline_once() -> Dict[str, Any]:
    cmd = [sys.executable, str(Path(__file__).resolve().parent.parent / "run_pipeline.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-2000:],
    }


def _video_watch_loop():
    global _LAST_VIDEO_COUNT, _AUTO_PIPELINE_RUNNING
    try:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        _LAST_VIDEO_COUNT = len([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
    except Exception:
        _LAST_VIDEO_COUNT = 0

    while True:
        try:
            time.sleep(8)
            current = len([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
            if current > _LAST_VIDEO_COUNT and not _AUTO_PIPELINE_RUNNING:
                with _PIPELINE_LOCK:
                    _AUTO_PIPELINE_RUNNING = True
                    _run_pipeline_once()
                    _AUTO_PIPELINE_RUNNING = False
            _LAST_VIDEO_COUNT = current
        except Exception:
            time.sleep(8)


def _delete_generated_files_for_video(video_name: str) -> Dict[str, Any]:
    deleted = []

    candidates = [
        TRANSCRIPT_DIR / f"{video_name}.json",
        TRANSCRIPT_DIR / f"{video_name}.mm.json",
        VISUAL_CAPTION_DIR / f"{video_name}.json",
    ]

    for p in candidates:
        try:
            if p.exists():
                p.unlink()
                deleted.append(str(p))
        except Exception:
            pass

    kf_dir = KEYFRAME_DIR / video_name
    try:
        if kf_dir.exists() and kf_dir.is_dir():
            shutil.rmtree(kf_dir)
            deleted.append(str(kf_dir))
    except Exception:
        pass

    return {"deleted_files": deleted, "deleted_files_count": len(deleted)}


@app.post("/api/rebuild/video")
def rebuild_video(req: RebuildVideoReq):
    video_name = (req.video_name or "").strip()
    if not video_name:
        raise HTTPException(status_code=400, detail="video_name 不能为空")

    result = {
        "video_name": video_name,
        "deleted_quiz": 0,
        "deleted_vector": False,
        "deleted_files": [],
        "deleted_files_count": 0,
    }

    if req.delete_quiz:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM quizzes WHERE video_name = ?", (video_name,))
            result["deleted_quiz"] = cursor.rowcount
            conn.commit()
            conn.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"删除题库失败: {e}")

    if req.delete_vector:
        if vectorstore is None:
            raise HTTPException(status_code=503, detail=f"向量数据库不可用: {vectorstore_error}")
        try:
            if hasattr(vectorstore, "delete"):
                vectorstore.delete(where={"video_name": video_name})
            else:
                vectorstore._collection.delete(where={"video_name": video_name})
            result["deleted_vector"] = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"删除向量数据失败: {e}")

    if req.delete_generated_files:
        file_result = _delete_generated_files_for_video(video_name)
        result.update(file_result)

    return result


@app.post("/api/pipeline/run")
def run_pipeline_manual(req: PipelineRunReq):
    cmd = [sys.executable, str(Path(__file__).resolve().parent.parent / "run_pipeline.py")]
    if req.continue_on_error:
        cmd.append("--continue-on-error")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"执行失败: {e}")
