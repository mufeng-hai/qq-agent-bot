# -*- coding: utf-8 -*-
"""
retrieve.py — 检索验证：问题 → embedding → Chroma 相似度查询

用法：
  python rag/retrieve.py "群里能点外卖吗"
  python rag/retrieve.py "你是不是AI" 3      # 第二个参数可指定 top-k，默认 3

判定规则：
  similarity = 1 - distance（Chroma cosine 距离转相似度）
  >= SIM_THRESHOLD 视为命中（0.5 起调，凭实测 badcase 逐步上调，过高漏检、过低误检）

注意：查询侧没加 bge 官方推荐的 instruction 前缀，与 ingest 侧编码保持一致；
若实测召回差，可实验「查询加 instruction、文档不加」的非对称编码（bge 官方用法）。
"""
import os
import sys
from pathlib import Path

import chromadb
import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "knowledge"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
SIM_THRESHOLD = 0.5   # 相似度阈值：0.5 起调

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")


def embed_one(text: str) -> list[float]:
    resp = httpx.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "input": [text]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def main() -> None:
    if not API_KEY:
        sys.exit("[x] 缺 SILICONFLOW_API_KEY（.env）")
    if len(sys.argv) < 2:
        sys.exit('用法: python rag/retrieve.py "问题" [top_k]')

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    qvec = embed_one(query)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    if col.count() == 0:
        sys.exit("[x] 向量库为空，先跑 python rag/ingest.py")

    res = col.query(query_embeddings=[qvec], n_results=top_k,
                    include=["documents", "metadatas", "distances"])
    print(f"问题：{query}\n")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1 - dist   # cosine 距离 → 相似度
        hit = "✓ 命中" if sim >= SIM_THRESHOLD else "✗ 未达阈值"
        preview = doc.replace("\n", " ")[:60]
        print(f"[{hit} {sim:.3f}] {meta['source']}#chunk{meta['chunk']}")
        print(f"    {preview}...\n")


if __name__ == "__main__":
    main()
