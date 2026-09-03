import os
from pathlib import Path
from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

# 自动加载项目根目录 .env
load_dotenv(ROOT_DIR / ".env")


def _path_from_env(key: str, default: Path) -> Path:
    raw = os.getenv(key)
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else (ROOT_DIR / p)


# 基础路径（相对路径统一按项目根目录解析）
VIDEO_DIR = _path_from_env("VIDEO_DIR", ROOT_DIR / "videos")
TRANSCRIPT_DIR = _path_from_env("TRANSCRIPT_DIR", ROOT_DIR / "transcripts")
SUPPLEMENTARY_DOCS_DIR = _path_from_env("SUPPLEMENTARY_DOCS_DIR", ROOT_DIR / "supplementary_docs")
CHROMA_DB_DIR = _path_from_env("CHROMA_DB_DIR", ROOT_DIR / "chroma_db")
DB_PATH = _path_from_env("DB_PATH", ROOT_DIR / "tutor_database.db")

# 模型与接口
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")

# 实时问答/评估（低延迟）
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
EVALUATE_MODEL_NAME = os.getenv("EVALUATE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# 离线题目生成（高质量）
QUIZ_MODEL_NAME = os.getenv("QUIZ_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")

# 视觉模型（图表/公式）
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")

# 兼容旧配置
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", CHAT_MODEL_NAME)

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", r"D:\AI_Models\AI-ModelScope\bge-small-zh-v1.5")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", r"D:\AI_Models\faster-whisper-base")

# 多模态处理配置
ENABLE_VLM = os.getenv("ENABLE_VLM", "true").lower() in ("1", "true", "yes", "on")
KEYFRAME_DIR = _path_from_env("KEYFRAME_DIR", ROOT_DIR / "keyframes")
VISUAL_CAPTION_DIR = _path_from_env("VISUAL_CAPTION_DIR", ROOT_DIR / "visual_captions")
KEYFRAME_MIN_INTERVAL_SEC = float(os.getenv("KEYFRAME_MIN_INTERVAL_SEC", "2"))
KEYFRAME_DIFF_THRESHOLD = float(os.getenv("KEYFRAME_DIFF_THRESHOLD", "18"))
KEYFRAME_MAX_PER_VIDEO = int(os.getenv("KEYFRAME_MAX_PER_VIDEO", "300"))
MM_MERGE_WINDOW_SEC = float(os.getenv("MM_MERGE_WINDOW_SEC", "3"))

# 服务配置
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ALLOW_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "null,http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if o.strip()
]
