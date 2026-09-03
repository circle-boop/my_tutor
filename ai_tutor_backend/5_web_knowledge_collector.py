import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any
import urllib.parse
from openai import OpenAI
from tavily import TavilyClient

try:
    from .config import (
        SUPPLEMENTARY_DOCS_DIR,
        VIDEO_DIR,
        LLM_API_KEY,
        LLM_BASE_URL,
        CHAT_MODEL_NAME,
        TRANSCRIPT_DIR,
        DB_PATH,
    )
except ImportError:
    from config import (
        SUPPLEMENTARY_DOCS_DIR,
        VIDEO_DIR,
        LLM_API_KEY,
        LLM_BASE_URL,
        CHAT_MODEL_NAME,
        TRANSCRIPT_DIR,
        DB_PATH,
    )

# 设置 Tavily API Key
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "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")

class WebSearchAgent:
    def __init__(self):
        self.llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.logs_dir = SUPPLEMENTARY_DOCS_DIR / "_agent_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.preferred_domains = [
            "wikipedia.org", "baike.baidu.com", "edu.cn", "arxiv.org",
            "tensorflow.org", "pytorch.org", "docs.python.org",
        ]
        self.low_priority_domains = [
            "cnblogs.com", "sohu.com", "toutiao.com",
        ]
        self.blocked_domains = [
            "smzdm.com", "csdn.net", "jingyan.baidu.com",
        ]
        
        # 统一提升生成文档的底线门槛
        self.min_high_quality_items = 3
        self.min_unique_domains = 2

    @staticmethod
    def _strip_video_prefix(video_name: str) -> str:
        return re.sub(r"^\s*\d+[\.、\-\s]*", "", (video_name or "").strip())

    def _safe_llm_json(self, prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm_client is None:
            return fallback
        try:
            response = self.llm_client.chat.completions.create(
                model=CHAT_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            txt = (response.choices[0].message.content or "").strip()
            if txt.startswith("```json"):
                txt = txt[7:].strip()
            if txt.startswith("```"):
                txt = txt[3:].strip()
            if txt.endswith("```"):
                txt = txt[:-3].strip()
            return json.loads(txt)
        except Exception:
            return fallback

    @staticmethod
    def _normalize_text(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    def _is_url_allowed(self, url: str) -> bool:
        u = (url or "").lower()
        if not u.startswith("http"):
            return False
        if u.endswith(".pdf"):
            return False
        return not any(d in u for d in self.blocked_domains)

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        base = re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,}", text or "")
        stopwords = {
            "什么是", "什么", "如何", "有哪些", "领域", "案例", "误区", "工作", "机制", "定义", "基本", "经典", "本节课",
            "the", "and", "for", "with", "what", "how",
        }
        out = []
        for w in base:
            w2 = w.strip().lower()
            if not w2 or w2 in stopwords:
                continue
            out.append(w2)
        return list(dict.fromkeys(out))

    def _is_content_relevant(self, content: str, query: str, topic: str) -> bool:
        text = self._normalize_text(content)
        if len(text) < 120:
            return False

        hard_negative = ["党员", "党务", "组织关系", "活动规则", "购物车", "登录后查看", "推广"]
        if any(k in text for k in hard_negative):
            return False

        keys = list(dict.fromkeys(self._extract_keywords(query) + self._extract_keywords(topic)))
        if not keys:
            return True

        hit = sum(1 for k in keys if k in text)
        if hit == 0:
            return False

        ai_markers = ["人工智能", "ai", "machine learning", "机器学习", "深度学习"]
        if any(m in self._normalize_text(query + " " + topic) for m in ai_markers):
            return any(m in text for m in ai_markers)
        return True

    @staticmethod
    def _merge_items(existing: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_url: Dict[str, Dict[str, Any]] = {}
        for item in existing + new_items:
            url = item.get("url", "")
            if not url:
                continue
            prev = by_url.get(url)
            if prev is None or item.get("quality_score", 0) > prev.get("quality_score", 0):
                by_url[url] = item
        merged = list(by_url.values())
        merged.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        return merged

    @staticmethod
    def _build_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            return {
                "total_items": 0,
                "avg_quality": 0.0,
                "high_quality_items": 0,
                "unique_domains": 0,
            }
        avg_score = round(sum(x.get("quality_score", 0) for x in items) / len(items), 3)
        domains = set(urllib.parse.urlparse(x.get("url", "")).netloc for x in items if x.get("url"))
        return {
            "total_items": len(items),
            "avg_quality": avg_score,
            "high_quality_items": sum(1 for x in items if x.get("quality_score", 0) >= 0.65),
            "unique_domains": len([d for d in domains if d]),
        }

    def _can_generate_doc(self, stats: Dict[str, Any]) -> bool:
        return (
            stats.get("high_quality_items", 0) >= self.min_high_quality_items
            and stats.get("unique_domains", 0) >= self.min_unique_domains
        )

    def _build_blocked_markdown(self, video_name: str, items: List[Dict[str, Any]], stats: Dict[str, Any], reason: str) -> str:
        lines = [
            f"# {video_name} 补充资料（待完善）",
            "",
            "## 当前状态",
            f"未生成正式补充文档：{reason}",
            "",
            "## 可信度门槛",
            f"- 高质量来源数 >= {self.min_high_quality_items}",
            f"- 不同域名数 >= {self.min_unique_domains}",
            "",
            "## 本次统计",
            f"- 抓取总条目: {stats.get('total_items', 0)}",
            f"- 高质量条目: {stats.get('high_quality_items', 0)}",
            f"- 覆盖域名数: {stats.get('unique_domains', 0)}",
            f"- 平均质量分: {stats.get('avg_quality', 0)}",
            "",
            "## 已检索来源（供人工复核）",
        ]
        for i, item in enumerate(items[:10], start=1):
            lines.append(
                f"- [{i}] {item.get('url', '')} | score={item.get('quality_score', 0)} | source={item.get('source', '')}"
            )
        return "\n".join(lines)

    def get_missing_videos(self) -> List[str]:
        if not VIDEO_DIR.exists():
            return []
        videos = [f.stem for f in VIDEO_DIR.glob('*.mp4')]
        if not SUPPLEMENTARY_DOCS_DIR.exists():
            SUPPLEMENTARY_DOCS_DIR.mkdir(parents=True)
        existing_docs = [f.stem for f in SUPPLEMENTARY_DOCS_DIR.glob('*.md')]
        missing = []
        for v in videos:
            if v not in existing_docs and f"{v}_基础知识" not in existing_docs:
                missing.append(v)
        return missing

    def determine_search_queries(self, video_name: str) -> Dict[str, Any]:
        knowledge_points = []
        try:
            import sqlite3
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT knowledge_point FROM quizzes WHERE video_name = ?", (video_name,))
            rows = cursor.fetchall()
            conn.close()
            knowledge_points = [row[0] for row in rows if row[0]]
        except Exception as e:
            print(f"[{video_name}] 读取知识点失败: {e}")

        topic = self._strip_video_prefix(video_name) or video_name
        context_prompt = ""
        if knowledge_points:
            kp_text = "\n".join(f"- {kp}" for kp in knowledge_points)
            context_prompt = f"以下是系统事先从该视频中提取的核心知识点串联：\n{kp_text}\n请分析这些知识点，理出这节课的核心知识体系。"
        else:
            context_prompt = "暂无知识点信息，请直接根据视频名称推断。"

        prompt = f"""
        你是课程知识搜集 Agent 的“规划器”。请基于课程主题，为视频生成检索计划。
        当前视频名：{video_name}
        主题定调词：{topic}
        {context_prompt}

        要求：
        1) 返回 3-5 个查询词，必须包含主题词，不可偏题。
        2) 为每个查询词给一个子目标（例如定义、机制、案例、误区）。
        3) 返回 JSON，不要 Markdown。

        JSON 格式：
        {{
          "topic": "课程主题",
          "queries": [
            {{"query": "查询词", "goal": "该查询想补充的知识"}}
          ]
        }}
        """
        fallback = {
            "topic": topic,
            "queries": [
                {"query": f"{topic} 基础概念", "goal": "建立概念定义"},
                {"query": f"{topic} 工作原理", "goal": "补充底层机制"},
                {"query": f"{topic} 应用案例", "goal": "补充实际应用"},
            ],
        }
        plan = self._safe_llm_json(prompt, fallback=fallback)
        items = plan.get("queries") if isinstance(plan, dict) else None
        if not isinstance(items, list) or not items:
            return fallback
            
        cleaned = []
        seen_queries = set()
        for it in items:
            if isinstance(it, dict) and it.get("query"):
                q = str(it.get("query", "")).strip()
                if not q:
                    continue
                topic_hint = str(plan.get("topic", topic)).strip()
                if topic_hint and topic_hint not in q:
                    q = f"{topic_hint} {q}".strip()
                q_norm = self._normalize_text(q)
                if q_norm in seen_queries:
                    continue
                seen_queries.add(q_norm)
                cleaned.append({"query": q, "goal": str(it.get("goal", "")).strip() or "补充课程知识"})
                
        return {"topic": plan.get("topic", topic), "queries": cleaned[:5]}

    def _domain_bonus(self, url: str) -> float:
        domain_weights = {
            "wikipedia.org": 0.25,
            "zhihu.com": 0.08,
            "baike.baidu.com": 0.22,
            "edu.cn": 0.2,
            "mooc": 0.12,
            "tensorflow.org": 0.2,
            "pytorch.org": 0.2,
            "arxiv.org": 0.18,
        }
        for domain, w in domain_weights.items():
            if domain in (url or ""):
                return w
        if any(d in (url or "") for d in self.low_priority_domains):
            return -0.08
        return 0.0

    @staticmethod
    def _keyword_overlap_score(text: str, keywords: List[str]) -> float:
        if not text or not keywords:
            return 0.0
        t = text.lower()
        hit = sum(1 for k in keywords if k and k.lower() in t)
        return min(1.0, hit / max(1, len(keywords)))

    def _score_chunk(self, text: str, url: str, query: str, topic: str) -> float:
        length_score = min(1.0, len(text) / 2000.0)
        overlap = self._keyword_overlap_score(text, [query, topic])
        domain = self._domain_bonus(url)
        noise_penalty = 0.0
        low_value_tokens = ["广告", "立即下载", "优惠", "活动规则", "购物车", "登录后查看"]
        if any(tok in text for tok in low_value_tokens):
            noise_penalty = 0.15
        return round(max(0.0, min(1.0, 0.45 * overlap + 0.4 * length_score + 0.15 + domain - noise_penalty)), 3)

    def _search_and_fetch_tavily(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """调用 Tavily API 获取高级检索和正文"""
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_raw_content=True
            )
            results = []
            for item in response.get("results", []):
                # 优先获取原始正文，无正文则用摘要兜底
                content = item.get("raw_content") or item.get("content")
                if not content:
                    continue
                results.append({
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "snippet": item.get("content"),
                    "content": content
                })
            return results
        except Exception as e:
            print(f"      [WARN] Tavily 检索失败 {query}: {e}")
            return []

    def perform_search_and_scrape(
        self, plan: Dict[str, Any], max_urls_per_query: int = 5, global_visited_urls: set = None
    ) -> Dict[str, Any]:
        topic = plan.get("topic", "")
        plan_items = plan.get("queries", [])
        results: List[Dict[str, Any]] = []
        visited_urls = global_visited_urls if global_visited_urls is not None else set()
        seen_text_fingerprints = set()
        round_seen_queries = set()

        for item in plan_items:
            query = (item.get("query") or "").strip()
            goal = (item.get("goal") or "").strip()
            if not query:
                continue

            q_norm = self._normalize_text(query)
            if q_norm in round_seen_queries:
                continue
            round_seen_queries.add(q_norm)

            print(f"    🔍 搜索: {query} | 目标: {goal}")
            
            # 使用 Tavily 替代原有的 Bing + 多线程抓取
            candidates = self._search_and_fetch_tavily(query, max_results=max_urls_per_query)
            
            for cand in candidates:
                url = cand.get("url", "")
                if not url or url in visited_urls:
                    continue
                if not self._is_url_allowed(url):
                    continue
                visited_urls.add(url)

                text = cand.get("content", "")
                if not self._is_content_relevant(text, query=query, topic=topic):
                    continue
                if len(text) < 350:
                    continue
                    
                text_fp = hash(self._normalize_text(text[:400]))
                if text_fp in seen_text_fingerprints:
                    continue
                seen_text_fingerprints.add(text_fp)
                
                score = self._score_chunk(text=text, url=url, query=query, topic=topic)
                if score < 0.5:
                    continue
                    
                print(f"      [OK] 捕获优质来源: {url[:64]}... (得分: {score})")
                results.append({
                    "query": query,
                    "goal": goal,
                    "source": "tavily_api",
                    "url": url,
                    "quality_score": score,
                    "content": text[:5000], # 截断超长文本保护 Token
                    "title": cand.get("title", ""),
                    "snippet_seed": cand.get("snippet", ""),
                })

            # 控制 API 请求频率
            time.sleep(0.5)

        results.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        return {
            "topic": topic,
            "items": results,
            "stats": self._build_stats(results),
        }

    def reflect_and_refine_plan(self, video_name: str, plan: Dict[str, Any], exec_result: Dict[str, Any], round_id: int) -> Dict[str, Any]:
        stats = exec_result.get("stats", {})
        enough = (
            stats.get("high_quality_items", 0) >= self.min_high_quality_items
            and stats.get("unique_domains", 0) >= self.min_unique_domains
            and stats.get("total_items", 0) >= 4
        )
        if enough:
            return {"need_retry": False, "reason": "资料质量与数量已达标", "new_plan": plan}

        topic = plan.get("topic") or self._strip_video_prefix(video_name)
        fallback_queries = [
            {"query": f"{topic} 核心定义 对比", "goal": "补充定义与边界"},
            {"query": f"{topic} 常见误区 纠偏", "goal": "补充误区辨析"},
            {"query": f"{topic} 真实应用 场景", "goal": "补充工程案例"},
        ]

        prompt = f"""
        你是知识搜集 Agent 的反思器。当前第 {round_id} 轮检索结果不足。
        视频：{video_name}
        当前主题：{topic}
        统计：{json.dumps(stats, ensure_ascii=False)}

        请输出新的查询计划（3-4条），要求：
        1) 更偏向补齐缺口（定义/机制/案例/误区）；
        2) 查询词必须带课程主题词；
        3) 仅输出 JSON：{{"queries":[{{"query":"...","goal":"..."}}]}}
        """
        out = self._safe_llm_json(prompt, fallback={"queries": fallback_queries})
        items = out.get("queries") if isinstance(out, dict) else None
        if not isinstance(items, list) or not items:
            items = fallback_queries

        normalized = []
        for it in items:
            if isinstance(it, dict) and it.get("query"):
                normalized.append({
                    "query": str(it.get("query", "")).strip(),
                    "goal": str(it.get("goal", "")).strip() or "补齐课程缺口",
                })
        if not normalized:
            normalized = fallback_queries

        return {
            "need_retry": True,
            "reason": "检索结果不足，建议重检",
            "new_plan": {"topic": topic, "queries": normalized[:4]},
        }

    def summarize_to_markdown(self, video_name: str, exec_result: Dict[str, Any]) -> str:
        items = exec_result.get("items", [])
        if not items:
            return f"# {video_name} 补充资料\n\n未能在互联网抓取到有效资料。"

        top_items = items[:8]
        reference_blocks = []
        for idx, item in enumerate(top_items, start=1):
            text = (item.get("content") or "")[:1200]
            reference_blocks.append(
                f"[{idx}] URL: {item.get('url', '')}\n"
                f"来源类型: {item.get('source', '')}\n"
                f"质量分: {item.get('quality_score', 0)}\n"
                f"内容摘录:\n{text}"
            )
        ref_blob = "\n\n-----\n\n".join(reference_blocks)

        # 修复提示词幻觉：严格约束根据资料生成，杜绝无中生有
        prompt = f"""
        你是课程知识库构建助手。请严格【仅基于】下述“带引用编号的资料”，为视频《{video_name}》撰写补充文档。

        【参考资料】
        {ref_blob}

        【输出要求】
        1. 必须包含章节：核心概念、底层机制、与本节课联系、应用案例、易错点。
        2. 忠实于参考资料：每个观点必须有资料支撑，并在句末标出对应的引用编号（如 [1]）。
        3. 严格禁止捏造：如果提供的参考资料中没有涵盖某个章节所需的信息，请直接在该章节下回复“当前检索到的资料不足以支撑该部分内容”，绝对禁止凭空捏造内容或引用编号。
        4. 直接输出 Markdown 正文，文末新增“## 参考来源”章节，按引用顺序列出 URL。
        """

        print(f"    [INFO] 正在利用 LLM 提炼总结，准备撰写深度知识库...")
        try:
            if self.llm_client is None:
                lines = [f"# {video_name} 补充资料", "", "## 参考来源"]
                for i, item in enumerate(top_items, start=1):
                    lines.append(f"- [{i}] {item.get('url', '')} (质量分: {item.get('quality_score', 0)})")
                return "\n".join(lines)

            response = self.llm_client.chat.completions.create(
                model=CHAT_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, # 调低温度以强化事实性
            )
            md_content = (response.choices[0].message.content or "").strip()
            if md_content.startswith("```markdown"):
                md_content = md_content[11:].strip()
            if md_content.startswith("```"):
                md_content = md_content[3:].strip()
            if md_content.endswith("```"):
                md_content = md_content[:-3].strip()
            return md_content
        except Exception as e:
            return f"# {video_name} 补充资料\n\n资料提炼失败：{e}"

    def _write_agent_log(self, video_name: str, payload: Dict[str, Any]) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.logs_dir / f"{video_name}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def run_agent(self):
        missing_videos = self.get_missing_videos()
        if not missing_videos:
            print("[OK] 所有视频均已有补充资料，Agent 无需执行。")
            return

        print(f"🔍 发现 {len(missing_videos)} 个视频缺少补充资料: {missing_videos}")
        
        for video_name in missing_videos:
            print(f"\n=============================================")
            print(f"[INFO] 开始为视频《{video_name}》构建网络补充资料...")

            logs: Dict[str, Any] = {"video_name": video_name, "rounds": []}
            global_visited_urls = set()
            cumulative_items: List[Dict[str, Any]] = []
            stop_reason = ""

            plan = self.determine_search_queries(video_name)
            print(f"  📝 首轮计划: {[q['query'] for q in plan.get('queries', [])]}")

            # 提高最大重试轮次和容错耐心值
            max_rounds = 4 
            no_gain_patience = 2 
            no_gain_rounds = 0
            final_exec = {"items": [], "stats": self._build_stats([])}
            
            for round_id in range(1, max_rounds + 1):
                print(f"  [INFO] 执行第 {round_id}/{max_rounds} 轮检索...")
                exec_result = self.perform_search_and_scrape(plan, global_visited_urls=global_visited_urls)
                
                prev_count = len(cumulative_items)
                cumulative_items = self._merge_items(cumulative_items, exec_result.get("items", []))
                cumulative_stats = self._build_stats(cumulative_items)
                gained = len(cumulative_items) - prev_count
                
                if gained <= 0:
                    no_gain_rounds += 1
                else:
                    no_gain_rounds = 0

                logs["rounds"].append({
                    "round": round_id,
                    "plan": plan,
                    "round_stats": exec_result.get("stats", {}),
                    "stats": cumulative_stats,
                    "new_items": max(0, gained),
                    "top_items": [{
                        "url": x.get("url"),
                        "quality_score": x.get("quality_score"),
                        "source": x.get("source"),
                        "query": x.get("query"),
                        "goal": x.get("goal"),
                        "snippet": (x.get("content") or "")[:200],
                    } for x in cumulative_items[:8]],
                })
                final_exec = {"items": cumulative_items, "stats": cumulative_stats}

                if round_id >= 2 and no_gain_rounds >= no_gain_patience:
                    stop_reason = f"连续{no_gain_rounds}轮无增量，提前停止"
                    print(f"  [WARN] 结束检索: {stop_reason}。")
                    break

                reflect = self.reflect_and_refine_plan(video_name, plan, final_exec, round_id=round_id)
                if not reflect.get("need_retry", False):
                    stop_reason = reflect.get("reason", "资料质量与数量已达标")
                    print(f"  [OK] 结束检索: {stop_reason}。")
                    break
                    
                if round_id == max_rounds:
                    stop_reason = "达到最大轮次，当前结果仍未完美达标"
                    print(f"  [WARN] 结束检索: {stop_reason}。")
                    break
                    
                print(f"  [INFO] 触发反思重检: {reflect.get('reason', '检索结果不足，建议重检')}")
                plan = reflect.get("new_plan", plan)

            generation_blocked = False
            generation_reason = ""
            final_stats = final_exec.get("stats", {})
            
            # 严格根据统一的质量阈值拦截生成
            if self._can_generate_doc(final_stats):
                md_content = self.summarize_to_markdown(video_name, final_exec)
            else:
                generation_blocked = True
                generation_reason = (
                    "未达到多源可信门槛 "
                    f"(要求 high_quality_items>={self.min_high_quality_items}, unique_domains>={self.min_unique_domains} | "
                    f"实际 high_quality_items={final_stats.get('high_quality_items', 0)}, unique_domains={final_stats.get('unique_domains', 0)})"
                )
                print(f"  🛑 拦截文档生成：{generation_reason}")
                md_content = self._build_blocked_markdown(
                    video_name=video_name,
                    items=final_exec.get("items", []),
                    stats=final_stats,
                    reason=generation_reason,
                )

            save_path = SUPPLEMENTARY_DOCS_DIR / f"{video_name}.md"
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            logs["final_stats"] = final_exec.get("stats", {})
            logs["stop_reason"] = stop_reason or "流程完成"
            logs["generation_blocked"] = generation_blocked
            logs["generation_reason"] = generation_reason
            logs["generation_threshold"] = {
                "min_high_quality_items": self.min_high_quality_items,
                "min_unique_domains": self.min_unique_domains,
            }
            logs["saved_doc"] = str(save_path)
            log_path = self._write_agent_log(video_name, logs)

            if not generation_blocked:
                print(f"[OK] 资料已成功生成并写入: {save_path}")
            print(f"🧾 Agent 日志已写入: {log_path}")

if __name__ == "__main__":
    agent = WebSearchAgent()
    agent.run_agent()
