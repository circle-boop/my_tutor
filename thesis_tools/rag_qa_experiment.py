import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from ai_tutor_backend.config import (
        TRANSCRIPT_DIR,
        SUPPLEMENTARY_DOCS_DIR,
        CHROMA_DB_DIR,
        EMBEDDING_MODEL_PATH,
        LLM_API_KEY,
        LLM_BASE_URL,
        CHAT_MODEL_NAME,
        DB_PATH,
    )
    from ai_tutor_backend.api_server import ChatReq, chat_with_rag
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ai_tutor_backend.config import (
        TRANSCRIPT_DIR,
        SUPPLEMENTARY_DOCS_DIR,
        CHROMA_DB_DIR,
        EMBEDDING_MODEL_PATH,
        LLM_API_KEY,
        LLM_BASE_URL,
        CHAT_MODEL_NAME,
        DB_PATH,
    )
    from ai_tutor_backend.api_server import ChatReq, chat_with_rag


QUESTION_SET: List[Dict[str, Any]] = [
    {
        "id": "Q01",
        "video_name": "1.什么是AI？",
        "question_type": "概念解释类",
        "current_time": 90.0,
        "question": "视频中是如何定义人工智能的？",
        "reference": "应说明 AI 是人类开发的程序，使计算机执行看似智能的任务，并和人类智能表现相似。",
    },
    {
        "id": "Q02",
        "video_name": "1.什么是AI？",
        "question_type": "视频上下文类",
        "current_time": 410.0,
        "question": "为什么视频中说机器狗和人互动可以作为判断 AI 的一个标准？",
        "reference": "应结合视频中 AI 五大特点，说明人机交互是判断智能的重要维度，能与环境或人进行交互的系统更接近 AI。",
    },
    {
        "id": "Q03",
        "video_name": "1.什么是AI？",
        "question_type": "总结归纳类",
        "current_time": 300.0,
        "question": "请总结《什么是AI？》这节视频的核心知识点。",
        "reference": "应覆盖人工智能定义、人工与智能含义、AI 五大特点，以及如何判断设备是否属于 AI。",
    },
    {
        "id": "Q04",
        "video_name": "2. 机器是怎么学习的？",
        "question_type": "概念解释类",
        "current_time": 210.0,
        "question": "机器学习为什么需要大量数据？",
        "reference": "应说明机器通过大量样本发现规律并进行预测，类似人类根据经验学习。",
    },
    {
        "id": "Q05",
        "video_name": "2. 机器是怎么学习的？",
        "question_type": "视频上下文类",
        "current_time": 260.0,
        "question": "视频中用手写图样识别说明了机器学习的什么过程？",
        "reference": "应指出图样识别依赖大量手写或图像数据训练，机器通过样本学习特征并进行判断。",
    },
    {
        "id": "Q06",
        "video_name": "2. 机器是怎么学习的？",
        "question_type": "总结归纳类",
        "current_time": 360.0,
        "question": "请概括《机器是怎么学习的？》主要讲了什么。",
        "reference": "应包含数据集、训练、预测、模式识别，以及机器学习和固定程序反应的区别。",
    },
    {
        "id": "Q07",
        "video_name": "3. 什么是算法？",
        "question_type": "概念解释类",
        "current_time": 70.0,
        "question": "什么是算法？它通常由哪些部分组成？",
        "reference": "应说明算法是解决问题或实现目标的一系列步骤，包含输入、处理步骤和输出。",
    },
    {
        "id": "Q08",
        "video_name": "3. 什么是算法？",
        "question_type": "视频上下文类",
        "current_time": 225.0,
        "question": "视频中让爸爸去开门的例子想说明编写算法时要注意什么？",
        "reference": "应说明算法步骤需要精确、明确、可执行，不能依赖含糊表达。",
    },
    {
        "id": "Q09",
        "video_name": "3. 什么是算法？",
        "question_type": "总结归纳类",
        "current_time": 150.0,
        "question": "请总结《什么是算法？》这节课的主要内容。",
        "reference": "应覆盖算法概念、输入处理输出、步骤明确性，以及调试 Debug 的意义。",
    },
    {
        "id": "Q10",
        "video_name": "4. 善于学习的机器",
        "question_type": "概念解释类",
        "current_time": 110.0,
        "question": "Teachable Machine 这类平台体现了机器学习的什么特点？",
        "reference": "应说明用户可通过采集样本、训练模型，让机器根据数据学习分类或识别规则。",
    },
    {
        "id": "Q11",
        "video_name": "4. 善于学习的机器",
        "question_type": "视频上下文类",
        "current_time": 320.0,
        "question": "视频中为什么要增加数据多样性或添加空类别？",
        "reference": "应说明增加数据多样性和空类别有助于减少误判，提高模型泛化和识别准确性。",
    },
    {
        "id": "Q12",
        "video_name": "4. 善于学习的机器",
        "question_type": "总结归纳类",
        "current_time": 220.0,
        "question": "请总结《善于学习的机器》中模型训练的大致流程。",
        "reference": "应包含采集样本、创建类别、训练模型、测试效果、调整数据和导出应用。",
    },
    {
        "id": "Q13",
        "video_name": "5. AI的偏见",
        "question_type": "概念解释类",
        "current_time": 180.0,
        "question": "机器学习中的偏见是如何形成的？",
        "reference": "应说明偏见来自数据集、学习算法和人的认知视角，训练数据有偏会导致模型输出有偏。",
    },
    {
        "id": "Q14",
        "video_name": "5. AI的偏见",
        "question_type": "视频上下文类",
        "current_time": 390.0,
        "question": "视频中人脸识别案例反映了 AI 偏见的什么问题？",
        "reference": "应说明人脸识别对部分群体识别效果较差，反映训练数据和技术系统可能固化社会偏见。",
    },
    {
        "id": "Q15",
        "video_name": "5. AI的偏见",
        "question_type": "总结归纳类",
        "current_time": 430.0,
        "question": "请总结《AI的偏见》这节视频的核心观点。",
        "reference": "应覆盖视角影响判断、数据和算法塑造机器认知、偏见会影响公平性和社会责任。",
    },
    {
        "id": "Q16",
        "video_name": "ANN基本原理",
        "question_type": "概念解释类",
        "current_time": 35.0,
        "question": "人工神经网络一般由哪些层组成？",
        "reference": "应说明人工神经网络包含输入层、隐藏层和输出层，人工神经元是基本单位。",
    },
    {
        "id": "Q17",
        "video_name": "ANN基本原理",
        "question_type": "视频上下文类",
        "current_time": 130.0,
        "question": "视频中神经网络通过训练提高预测准确度，主要依赖什么机制？",
        "reference": "应说明依赖误差计算、反向传播、梯度下降和权重调整。",
    },
    {
        "id": "Q18",
        "video_name": "ANN基本原理",
        "question_type": "总结归纳类",
        "current_time": 190.0,
        "question": "请概括 ANN 基本原理这节视频的知识框架。",
        "reference": "应覆盖神经元结构、网络层次、激活函数、误差平方和、学习率、反向传播和链式法则。",
    },
    {
        "id": "Q19",
        "video_name": "二次函数的图像和性质",
        "question_type": "概念解释类",
        "current_time": 60.0,
        "question": "二次函数 y=ax² 的图像有哪些基本性质？",
        "reference": "应说明图像是抛物线，顶点在原点，对称轴是 y 轴，开口方向与 a 的符号有关。",
    },
    {
        "id": "Q20",
        "video_name": "二次函数的图像和性质",
        "question_type": "视频上下文类",
        "current_time": 430.0,
        "question": "视频中如何利用抛物线的对称轴判断不同点的函数值大小？",
        "reference": "应说明先比较各点到对称轴的距离，再结合开口方向判断函数值大小。",
    },
    {
        "id": "Q21",
        "video_name": "二次函数的图像和性质",
        "question_type": "总结归纳类",
        "current_time": 300.0,
        "question": "请总结二次函数图像和性质这节课的主要内容。",
        "reference": "应覆盖抛物线顶点、对称轴、开口方向、顶点坐标求法，以及利用图像性质比较函数值。",
    },
    {
        "id": "Q22",
        "video_name": "神经网络原理",
        "question_type": "概念解释类",
        "current_time": 120.0,
        "question": "神经网络中的激活函数有什么作用？",
        "reference": "应说明激活函数引入非线性，使网络能够表达复杂关系或逻辑。",
    },
    {
        "id": "Q23",
        "video_name": "神经网络原理",
        "question_type": "视频上下文类",
        "current_time": 580.0,
        "question": "视频中提到 MNIST 数字识别时，神经网络输出层通常表示什么？",
        "reference": "应说明输出层通常对应不同数字类别的概率或预测结果。",
    },
    {
        "id": "Q24",
        "video_name": "神经网络原理",
        "question_type": "总结归纳类",
        "current_time": 500.0,
        "question": "请总结神经网络原理视频中的核心知识点。",
        "reference": "应覆盖神经元、前向传播、反向传播、损失函数、梯度下降、学习率和过拟合防止方法。",
    },
]


def format_time(seconds: float) -> str:
    sec = int(float(seconds))
    m, s = divmod(max(0, sec), 60)
    return f"{m}min{s}s" if m else f"{s}s"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_subtitles(video_name: str, multimodal: bool = False) -> List[Dict[str, Any]]:
    path = TRANSCRIPT_DIR / f"{video_name}.mm.json" if multimodal else TRANSCRIPT_DIR / f"{video_name}.json"
    if not path.exists():
        return []
    rows = load_json(path)
    chunks = []
    merge_count = 3
    for i in range(0, len(rows), merge_count):
        part = rows[i : i + merge_count]
        if not part:
            continue
        text_parts = []
        for item in part:
            tx = (item.get("text") or "").strip()
            if tx:
                text_parts.append(tx)
            if multimodal:
                for label, key in [("画面", "visual_text"), ("OCR", "ocr_text")]:
                    value = (item.get(key) or "").strip()
                    if value:
                        text_parts.append(f"[{label}] {value}")
                formulas = item.get("formula_latex") or []
                charts = item.get("chart_data") or []
                if formulas:
                    text_parts.append("[公式] " + "; ".join(str(x) for x in formulas if x))
                if charts:
                    text_parts.append("[图表] " + "; ".join(str(x) for x in charts if x))
        content = "\n".join(text_parts).strip()
        if content:
            chunks.append(
                {
                    "content": content,
                    "start": part[0].get("start", 0),
                    "end": part[-1].get("end", 0),
                }
            )
    return chunks


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve_subtitle_context(embeddings: HuggingFaceEmbeddings, cache: Dict[str, Any], video_name: str, query: str, k: int = 4) -> str:
    if video_name not in cache:
        chunks = chunk_subtitles(video_name, multimodal=False)
        texts = [x["content"] for x in chunks]
        vectors = embeddings.embed_documents(texts) if texts else []
        cache[video_name] = {"chunks": chunks, "vectors": vectors}
    entry = cache[video_name]
    if not entry["chunks"]:
        return "未检索到字幕资料。"
    qv = embeddings.embed_query(query)
    scored = []
    for chunk, vec in zip(entry["chunks"], entry["vectors"]):
        scored.append((cosine(qv, vec), chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for score, chunk in scored[:k]:
        lines.append(
            f"[{video_name}, {format_time(chunk['start'])}-{format_time(chunk['end'])}, 字幕, score={score:.3f}] {chunk['content']}"
        )
    return "\n".join(lines)


def load_supplementary_chunks(video_name: str) -> List[Dict[str, Any]]:
    candidates = [
        SUPPLEMENTARY_DOCS_DIR / f"{video_name}.md",
        SUPPLEMENTARY_DOCS_DIR / f"{video_name}_基础知识.md",
    ]
    doc_path = next((p for p in candidates if p.exists()), None)
    if not doc_path:
        return []
    text = doc_path.read_text(encoding="utf-8", errors="ignore")
    blocks = []
    current = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and current:
            blocks.append("\n".join(current).strip())
            current = [stripped]
        else:
            current.append(stripped)
    if current:
        blocks.append("\n".join(current).strip())
    if not blocks:
        blocks = [text]

    chunks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= 900:
            chunks.append({"content": block, "file_name": doc_path.name})
        else:
            for i in range(0, len(block), 700):
                chunks.append({"content": block[i : i + 900], "file_name": doc_path.name})
    return chunks


def retrieve_supplementary_context(embeddings: HuggingFaceEmbeddings, cache: Dict[str, Any], video_name: str, query: str, k: int = 2) -> str:
    key = f"supp::{video_name}"
    if key not in cache:
        chunks = load_supplementary_chunks(video_name)
        texts = [x["content"] for x in chunks]
        vectors = embeddings.embed_documents(texts) if texts else []
        cache[key] = {"chunks": chunks, "vectors": vectors}
    entry = cache[key]
    if not entry["chunks"]:
        return "未检索到补充资料。"
    qv = embeddings.embed_query(query)
    scored = [(cosine(qv, vec), chunk) for chunk, vec in zip(entry["chunks"], entry["vectors"])]
    scored.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for score, chunk in scored[:k]:
        lines.append(f"[{chunk['file_name']}, 补充资料, score={score:.3f}] {chunk['content']}")
    return "\n".join(lines)


def build_subtitle_time_window(video_name: str, current_time: float, radius: float = 30.0) -> str:
    path = TRANSCRIPT_DIR / f"{video_name}.json"
    if not path.exists():
        return "未找到当前视频字幕。"
    rows = load_json(path)
    start_window = float(current_time) - radius
    end_window = float(current_time) + radius
    lines = []
    for item in rows:
        if item.get("start", 0) <= end_window and item.get("end", 0) >= start_window:
            text = (item.get("text") or "").strip()
            if text:
                lines.append(
                    f"[{format_time(item.get('start', 0))}-{format_time(item.get('end', 0))}] {text}"
                )
    if not lines:
        return "当前时间窗内未检索到字幕。"
    return "\n".join(lines)


def llm_call(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    retries: int = 2,
    retry_sleep: float = 3.0,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=800,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = exc
            if attempt > retries:
                break
            wait_sec = retry_sleep * attempt
            print(
                f"  [WARN] LLM call failed ({type(exc).__name__}: {exc}); "
                f"retry {attempt}/{retries} after {wait_sec:.1f}s",
                flush=True,
            )
            time.sleep(wait_sec)
    raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_error}") from last_error


def plain_llm_answer(client: OpenAI, item: Dict[str, Any]) -> str:
    return llm_call(
        client,
        [
            {"role": "system", "content": "你是一个在线课程智能助教。请直接回答学生问题，表达清晰、简洁。"},
            {"role": "user", "content": item["question"]},
        ],
    )


def subtitle_rag_answer(client: OpenAI, embeddings: HuggingFaceEmbeddings, cache: Dict[str, Any], item: Dict[str, Any]) -> str:
    subtitle_context = retrieve_subtitle_context(embeddings, cache, item["video_name"], item["question"], k=3)
    supplementary_context = retrieve_supplementary_context(embeddings, cache, item["video_name"], item["question"], k=2)
    local_subtitle_context = build_subtitle_time_window(
        item["video_name"],
        float(item["current_time"]),
        radius=30.0,
    )
    prompt = f"""请基于【字幕检索片段】、【当前播放时间窗字幕】和【补充资料】回答学生问题。若资料中没有依据，请说明资料不足，不要编造。

【视频】{item['video_name']}
【当前时间】{format_time(item['current_time'])}
【问题】{item['question']}
【字幕检索片段】
{subtitle_context}

【当前播放时间窗字幕】
{local_subtitle_context}

【补充资料】
{supplementary_context}
"""
    return llm_call(
        client,
        [
            {"role": "system", "content": "你是一个基于课程字幕和补充资料回答问题的助教。"},
            {"role": "user", "content": prompt},
        ],
    )


def multimodal_rag_answer(item: Dict[str, Any]) -> str:
    req = ChatReq(
        query=item["question"],
        video_name=item["video_name"],
        current_time=float(item["current_time"]),
        history=[],
    )
    result = chat_with_rag(req)
    return result.get("answer", "")


def load_existing(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"items": []}


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


ANSWER_STRATEGIES = ["plain_llm", "subtitle_rag", "multimodal_rag"]


def has_all_answers(record: Dict[str, Any]) -> bool:
    answers = record.get("answers", {})
    return all(bool(answers.get(strategy)) for strategy in ANSWER_STRATEGIES)


def upsert_record(
    items: List[Dict[str, Any]],
    by_id: Dict[str, Dict[str, Any]],
    record: Dict[str, Any],
    raw_path: Path,
    data: Dict[str, Any],
) -> None:
    item_id = record["id"]
    if item_id in by_id:
        items[items.index(by_id[item_id])] = record
    else:
        items.append(record)
    by_id[item_id] = record
    save_json(raw_path, data)


def write_outputs(out_dir: Path, data: Dict[str, Any]) -> None:
    answer_rows = []
    score_rows = []
    for item in data["items"]:
        for strategy in ANSWER_STRATEGIES:
            answer = item.get("answers", {}).get(strategy, "")
            error = item.get("answer_errors", {}).get(strategy, "")
            answer_rows.append(
                {
                    "id": item["id"],
                    "video_name": item["video_name"],
                    "question_type": item["question_type"],
                    "current_time": item["current_time"],
                    "question": item["question"],
                    "reference": item["reference"],
                    "strategy": strategy,
                    "answer": answer,
                    "error": error,
                }
            )
            score_rows.append(
                {
                    "id": item["id"],
                    "video_name": item["video_name"],
                    "question_type": item["question_type"],
                    "strategy": strategy,
                    "relevance": "",
                    "accuracy": "",
                    "context_match": "",
                    "explainability": "",
                    "hallucination": "",
                    "notes": "",
                }
            )

    answers_csv = out_dir / "rag_qa_answers.csv"
    with open(answers_csv, "w", encoding="utf-8-sig", newline="") as f:
        if answer_rows:
            writer = csv.DictWriter(f, fieldnames=list(answer_rows[0].keys()))
            writer.writeheader()
            writer.writerows(answer_rows)

    scoring_csv = out_dir / "rag_qa_manual_scoring_template.csv"
    with open(scoring_csv, "w", encoding="utf-8-sig", newline="") as f:
        if score_rows:
            writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
            writer.writeheader()
            writer.writerows(score_rows)

    md = ["# 检索增强问答效果分析实验结果", ""]
    md.append("本文件仅记录问题集设计和答案生成情况。为避免生成模型自评造成偏差，评分应采用人工评分或不同模型盲评。")
    md.append("")
    md.append("## 问题集设计")
    md.append("")
    md.append("| 问题类型 | 数量 | 评价重点 |")
    md.append("| --- | ---: | --- |")
    md.append("| 概念解释类 | 8 | 概念定义是否准确，是否符合课程内容 |")
    md.append("| 视频上下文类 | 8 | 是否结合当前视频片段和局部时间点 |")
    md.append("| 总结归纳类 | 8 | 是否覆盖主要知识点，结构是否清晰 |")
    md.append("")
    md.append("## 输出文件")
    md.append("")
    md.append("- `rag_qa_experiment_raw.json`：原始问题和三组回答。")
    md.append("- `rag_qa_answers.csv`：便于查看和导入表格软件的三组回答。")
    md.append("- `rag_qa_manual_scoring_template.csv`：人工评分模板。")
    (out_dir / "rag_qa_experiment_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG QA comparison experiment.")
    parser.add_argument("--out-dir", default="thesis_outputs/rag_qa_experiment")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0, help="单次大模型 API 调用超时秒数")
    args = parser.parse_args()

    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is empty. Please configure .env first.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "rag_qa_experiment_raw.json"
    data = load_existing(raw_path)
    items = data.setdefault("items", [])
    by_id = {x["id"]: x for x in items}

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=args.timeout)
    print("[INFO] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
    subtitle_cache: Dict[str, Any] = {}

    questions = QUESTION_SET[: args.limit] if args.limit else QUESTION_SET
    for idx, item in enumerate(questions, 1):
        print(f"[RUN] {idx}/{len(questions)} {item['id']} {item['video_name']} {item['question_type']}")
        started = time.time()
        record = by_id.get(item["id"])
        if record:
            for key, value in item.items():
                record.setdefault(key, value)
            record.setdefault("answers", {})
            record.setdefault("answer_errors", {})
        else:
            record = {**item, "answers": {}, "answer_errors": {}}

        if has_all_answers(record):
            print(f"  [SKIP] all answers already exist", flush=True)
            continue

        for strategy in ANSWER_STRATEGIES:
            if record["answers"].get(strategy):
                print(f"  [SKIP] {strategy} answer exists", flush=True)
                continue

            print(f"  [ANSWER] {strategy}", flush=True)
            strategy_started = time.time()
            try:
                if strategy == "plain_llm":
                    answer = plain_llm_answer(client, item)
                elif strategy == "subtitle_rag":
                    answer = subtitle_rag_answer(client, embeddings, subtitle_cache, item)
                elif strategy == "multimodal_rag":
                    answer = multimodal_rag_answer(item)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                record["answers"][strategy] = answer
                record["answer_errors"].pop(strategy, None)
                record[f"{strategy}_elapsed_sec"] = round(time.time() - strategy_started, 2)
                print(
                    f"  [OK] {strategy} saved, elapsed={record[f'{strategy}_elapsed_sec']}s",
                    flush=True,
                )
            except Exception as exc:
                record["answer_errors"][strategy] = f"{type(exc).__name__}: {exc}"
                print(
                    f"  [WARN] {strategy} failed, saved error and continued: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            finally:
                record["elapsed_sec"] = round(time.time() - started, 2)
                upsert_record(items, by_id, record, raw_path, data)

        record["elapsed_sec"] = round(time.time() - started, 2)
        upsert_record(items, by_id, record, raw_path, data)
        print(f"[OK] {item['id']} saved, elapsed={record['elapsed_sec']}s")
        if args.sleep:
            time.sleep(args.sleep)

    write_outputs(out_dir, data)
    print(f"[OK] Experiment outputs saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
