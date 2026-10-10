# -*- coding: utf-8 -*-
"""
astrbot_plugin_rag — RAG 知识库检索增强

工作方式（挂在 AstrBot 的 LLM 请求钩子上，非侵入）：
  用户消息 → embedding → Chroma top-k 检索 → 相似度过阈值的片段注入 system prompt
  → LLM 基于「人设 + 知识片段」生成回答；未命中则什么都不做，走原有对话

三条设计原则：
  1. 非侵入：不改 AstrBot 本体、不改人设；停用插件即完全回滚
  2. 故障降级：检索链路任何异常只记日志、放行请求——RAG 挂了不能挂聊天
  3. 单一事实源：API key 与向量库都指向 qq-agent-bot 仓库，插件里不存第二份
"""
import os
from pathlib import Path

import chromadb
import httpx
from dotenv import load_dotenv

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core import logger

# ===== 配置常量（阶段 5 工程化时迁移到 AstrBot 插件配置面板）=====
KB_ROOT = Path(r"D:\.测试\机器人项目迁移\qq-agent-bot")   # 仓库根（.env + data/chroma）
COLLECTION = "knowledge"
SIM_THRESHOLD = 0.5    # 相似度阈值（retrieve.py 实测：相关 0.51-0.69，无关 0.41-0.48）
TOP_K = 3
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"


@register(
    "astrbot_plugin_rag",
    "mufeng-hai",
    "RAG 知识库检索增强",
    "LLM 请求前检索 Chroma 知识库，命中片段注入 system prompt",
    "0.1.0",
)
class RagPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        load_dotenv(KB_ROOT / ".env")
        self.api_key = os.getenv("SILICONFLOW_API_KEY", "")
        self.model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
        self.collection = None
        try:
            client = chromadb.PersistentClient(path=str(KB_ROOT / "data" / "chroma"))
            self.collection = client.get_or_create_collection(
                COLLECTION, metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"rag: 插件就绪，知识库共 {self.collection.count()} 条向量")
        except Exception as e:
            logger.error(f"rag: Chroma 初始化失败，插件将放行所有请求: {e}")

    async def _embed(self, text: str) -> list[float] | None:
        """查询向量化；失败返回 None（调用方据此放行）"""
        if not self.api_key:
            logger.warning("rag: 缺 SILICONFLOW_API_KEY，跳过检索")
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    EMBED_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": [text]},
                )
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"rag: embedding 调用失败，本次放行: {e}")
            return None

    @filter.on_llm_request()
    async def inject_knowledge(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求前注入知识：命中 → 追加 system_prompt；未命中/异常 → 放行"""
        try:
            query = (event.message_str or "").strip()
            if not query or self.collection is None:
                return
            if self.collection.count() == 0:
                return

            qvec = await self._embed(query)
            if qvec is None:
                return

            # 本地 ANN 查询（<10ms），同步调用可接受
            res = self.collection.query(
                query_embeddings=[qvec],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"],
            )
            hits = [
                (doc, meta, 1 - dist)   # cosine 距离 → 相似度
                for doc, meta, dist in zip(
                    res["documents"][0], res["metadatas"][0], res["distances"][0]
                )
                if 1 - dist >= SIM_THRESHOLD
            ]
            if not hits:
                logger.info(f"rag: 未命中 | {query[:30]}")
                return

            refs = "\n\n".join(
                f"【参考{i}】（来自 {m['source']}，相似度 {s:.2f}）\n{d}"
                for i, (d, m, s) in enumerate(hits, 1)
            )
            req.system_prompt = (
                (req.system_prompt or "")
                + "\n\n[知识库检索结果]\n以下片段与用户最新消息语义相关，回答时优先依据这些内容；"
                "若其中给出了明确口径（如标准应答、边界规则），请遵循，但保持你原本的说话风格。\n\n"
                + refs
            )
            logger.info(
                f"rag: 命中 {len(hits)} 条 | {query[:30]} | "
                f"top1={hits[0][2]:.3f} ({hits[0][1]['source']})"
            )
        except Exception as e:
            logger.warning(f"rag: 检索注入异常，已放行: {e}")
