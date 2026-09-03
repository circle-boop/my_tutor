# my_tutor（引导式智能助教）

一个本地可运行的「视频课程 + 随堂测验 + RAG 问答」助教系统。

## 1. 功能概览

- 视频播放过程中按时间点弹出选择题
- 基于字幕自动提炼知识点并出题
- 支持补充资料（pdf/txt/md/docx）向量化检索
- 聊天问答结合：
  - 向量库检索结果
  - 当前视频播放时间附近（±30s）局部上下文

## 2. 项目结构

- `index.html`：前端页面（Vue3）
- `ai_tutor_backend/api_server.py`：后端 API（FastAPI）
- `ai_tutor_backend/1_transcribe.py`：视频转字幕 + LLM 纠错
- `ai_tutor_backend/1_5_extract_keyframes_and_vlm.py`：关键帧提取 + VLM 图表/公式解析
- `ai_tutor_backend/1_6_merge_multimodal_timeline.py`：语音字幕与视觉信息时序融合
- `ai_tutor_backend/2_extract_knowledge.py`：知识点提炼 + 题库入库（多模态）
- `ai_tutor_backend/3_build_vector_db.py`：字幕/视觉融合向量化
- `ai_tutor_backend/4_add_supplementary_docs.py`：补充资料向量化
- `tutor_database.db`：SQLite 题库
- `chroma_db/`：Chroma 向量库

## 3. 环境准备

1) 安装依赖：

```bash
pip install -r requirements.txt
```

2) 配置环境变量：

```bash
copy .env.example .env
```

然后编辑 `.env`，至少填好：

- `LLM_API_KEY`
- `EMBEDDING_MODEL_PATH`
- `WHISPER_MODEL_PATH`
- `CHAT_MODEL_NAME`（建议 `Qwen/Qwen2.5-7B-Instruct`）
- `EVALUATE_MODEL_NAME`（建议 `Qwen/Qwen2.5-7B-Instruct`）
- `QUIZ_MODEL_NAME`（建议 `Qwen/Qwen2.5-14B-Instruct`）
- `VLM_MODEL_NAME`（建议 `Qwen/Qwen3-VL-8B-Instruct`）

## 4. 数据处理流程（首次或新增视频后）

在项目根目录执行：

```bash
python ai_tutor_backend/1_transcribe.py
python ai_tutor_backend/1_5_extract_keyframes_and_vlm.py
python ai_tutor_backend/1_6_merge_multimodal_timeline.py
python ai_tutor_backend/2_extract_knowledge.py
python ai_tutor_backend/3_build_vector_db.py
python ai_tutor_backend/5_web_knowledge_collector.py
python ai_tutor_backend/4_add_supplementary_docs.py
```

也可以一键自动执行整套流程（推荐）：

```bash
python run_pipeline.py
```

Windows 下可直接双击或命令行执行：

```bat
run_pipeline.bat
```

说明：默认某阶段失败会立即停止；如需失败后继续执行后续阶段：

```bash
python run_pipeline.py --continue-on-error
```

### 知识搜集 Agent（最小版）

`ai_tutor_backend/5_web_knowledge_collector.py` 现已按最小 Agent 流程执行：

- 规划：基于视频主题和题库知识点生成检索计划（查询词 + 子目标）
- 执行：多源抓取（Bing 网页 + Wikipedia API）并为每条来源计算质量分
- 反思：当资料数量/质量不足时自动二次检索（最多 1-2 轮重试）
- 产出：生成带引用编号的 `supplementary_docs/*.md`，并输出日志到 `supplementary_docs/_agent_logs/*.json`

当前版本还包含两项质量与速度策略：

- 白名单优先：搜索结果会优先保留百科/高校/官方文档来源，低质量站点仅作为兜底
- 早停机制：若连续一轮无新增高质量来源，将提前停止后续轮次，避免空转等待

## 5. 启动后端

```bash
uvicorn ai_tutor_backend.api_server:app --host 127.0.0.1 --port 8000 --reload
```

## 6. 访问前端

直接用浏览器打开 `index.html`（或用本地静态服务器打开）。

> 注意：`index.html` 默认请求 `http://127.0.0.1:8000/api`。

## 7. 安全建议

- 不要把真实 API Key 提交到 Git
- `.env` 应加入 `.gitignore`
- 生产环境请把 `CORS_ALLOW_ORIGINS` 限制为你的前端域名
