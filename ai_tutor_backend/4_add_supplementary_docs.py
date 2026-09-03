import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from .config import SUPPLEMENTARY_DOCS_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_PATH
except ImportError:
    from config import SUPPLEMENTARY_DOCS_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL_PATH


def add_supplementary_docs():
    print("正在加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    vectorstore = Chroma(persist_directory=str(CHROMA_DB_DIR), embedding_function=embeddings)

    existing_data = vectorstore.get(include=["metadatas"])
    processed_files = set()
    if existing_data and existing_data["metadatas"]:
        for meta in existing_data["metadatas"]:
            if "file_name" in meta:
                processed_files.add(meta["file_name"])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    new_docs = []

    for filename in os.listdir(SUPPLEMENTARY_DOCS_DIR):
        if not filename.endswith((".pdf", ".txt", ".md", ".docx")):
            continue

        if filename in processed_files:
            print(f"跳过: 文件 [{filename}] 已在向量库中。")
            continue

        file_path = SUPPLEMENTARY_DOCS_DIR / filename
        print(f"正在读取并切分: {filename}")

        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(str(file_path))
                docs_from_file = loader.load()
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(str(file_path))
                docs_from_file = loader.load()
            else:
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs_from_file = loader.load()
        except Exception as e:
            print(f"读取文件 {filename} 失败: {e}")
            continue

        split_docs = text_splitter.split_documents(docs_from_file)

        for doc in split_docs:
            doc.metadata = {
                "source_type": "supplementary_doc",
                "file_name": filename,
            }
            new_docs.append(doc)

    if new_docs:
        vectorstore.add_documents(new_docs)
        print(f"\n补充资料更新完成！新增了 {len(new_docs)} 个文本块。")
    else:
        print("\n没有检测到需要处理的新补充资料。")


if __name__ == "__main__":
    add_supplementary_docs()
