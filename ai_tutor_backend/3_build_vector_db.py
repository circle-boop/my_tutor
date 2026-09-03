import os
import json
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from .config import TRANSCRIPT_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_PATH
except ImportError:
    from config import TRANSCRIPT_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_PATH


def build_vector_database():
    print("正在加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    vectorstore = Chroma(persist_directory=str(CHROMA_DB_DIR), embedding_function=embeddings)

    existing_data = vectorstore.get(include=["metadatas"])
    processed_videos = set()
    if existing_data and existing_data["metadatas"]:
        processed_videos = {meta["video_name"] for meta in existing_data["metadatas"] if "video_name" in meta}

    new_docs_count = 0

    for filename in os.listdir(TRANSCRIPT_DIR):
        if not filename.endswith(".json") or filename.endswith(".mm.json"):
            continue

        video_name = filename.replace(".json", "")

        if video_name in processed_videos:
            print(f"跳过: 视频 [{video_name}] 已在向量库中。")
            continue

        mm_path = TRANSCRIPT_DIR / f"{video_name}.mm.json"
        file_path = mm_path if mm_path.exists() else (TRANSCRIPT_DIR / filename)
        with open(file_path, "r", encoding="utf-8") as f:
            subtitles = json.load(f)

        docs = []
        MERGE_COUNT = 3
        for i in range(0, len(subtitles), MERGE_COUNT):
            chunk = subtitles[i:i + MERGE_COUNT]
            if not chunk:
                continue

            parts = []
            has_visual = False
            scene_types = set()
            formula_count = 0
            for item in chunk:
                t = (item.get("text") or "").strip()
                if t:
                    parts.append(t)
                v = (item.get("visual_text") or "").strip()
                if v:
                    parts.append(f"[视觉] {v}")
                    has_visual = True
                ocr = (item.get("ocr_text") or "").strip()
                if ocr:
                    parts.append(f"[OCR] {ocr}")
                    has_visual = True
                formulas = item.get("formula_latex") or []
                if formulas:
                    formula_count += len(formulas)
                    parts.append("[公式] " + " ; ".join([str(x) for x in formulas if x]))
                    has_visual = True
                charts = item.get("chart_data") or []
                if charts:
                    parts.append("[图表] " + " ; ".join([str(x) for x in charts if x]))
                    has_visual = True
                st = item.get("scene_type")
                if st:
                    scene_types.add(str(st))

            combined_text = "\n".join(parts)
            metadata = {
                "source_type": "video_multimodal" if has_visual else "video_subtitle",
                "video_name": video_name,
                "start_time": chunk[0].get("start", 0),
                "end_time": chunk[-1].get("end", 0),
                "has_visual": has_visual,
                "scene_type": ",".join(sorted(scene_types)) if scene_types else "unknown",
                "formula_count": formula_count,
            }
            docs.append(Document(page_content=combined_text, metadata=metadata))

        if docs:
            vectorstore.add_documents(docs)
            new_docs_count += len(docs)
            print(f" -> 视频 [{video_name}] 已成功向量化并入库 ({len(docs)} 个片段)。")

    if new_docs_count > 0:
        print(f"\n向量数据库更新完成！新增了 {new_docs_count} 个文本块。")
    else:
        print("\n没有检测到需要处理的新视频。")


if __name__ == "__main__":
    build_vector_database()
