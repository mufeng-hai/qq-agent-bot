# -*- coding: utf-8 -*-
"""
ingest.py — 知识库入库：knowledge/*.md → embedding → Chroma 向量库

管线：读 knowledge/ → 切片 → SiliconFlow embedding（bge-large-zh）→ upsert 进 Chroma（data/chroma）

四个设计决策（为什么这么写）：
  1. 切片：按 ## 小节切（保留语义完整性），超过 400 字再按段落累积二切——
     chunk 过大稀释向量表征、过小丢失上下文，且 bge 上下文 512 token（中文约一字一 token）
  2. 来源脚注剥离：md 末尾 "> 来源：..." 是给人看的溯源标注，对检索是噪声，不进向量
  3. 幂等：chunk id = hash(文件名+序号)，重跑本脚本是 upsert 按 id 覆盖——知识库可反复更新不重复
  4. cosine 距离：bge 系向量按余弦相似度训练，Chroma 默认 l2，不显式指定会算错相似度
"""
import hashlib
import os
import re
import sys
from pathlib import Path

import chromadb
import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent          # 仓库根（本脚本位于 rag/）
KB_DIR = ROOT / "knowledge"                            # 知识源（入库 git，是唯一事实源）
CHROMA_DIR = ROOT / "data" / "chroma"                  # 向量持久化（.gitignore 拦截 data/，向量是派生物）
COLLECTION_NAME = "knowledge"
CHUNK_MAX = 400                                        # 切片字数上限（对应 bge 512 token 上下文）
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
BATCH = 32                                             # SiliconFlow 每次 embedding 请求上限

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")


def load_chunks() -> list[dict]:
    """读取全部 md → 剥来源脚注 → 切片；返回 [{'id','text','meta'}]

    切片策略（v2，badcase 驱动的迭代）：
      - 问答对照行（`- 「...」→ ...`）每条独立成 chunk，并带上文件标题作上下文
        —— 短查询 vs 整卡片的相似度会被卡片其余内容稀释（实测 0.41-0.47 过不了 0.5 阈值，
           而闲聊题也有 0.40-0.48，两组分布重叠、阈值无解）；单条问答成 chunk 后
           问题原句占比 1/3，知识题分数显著抬升、闲聊题不变，分布被拉开
      - 其余段落照旧：按 ## 小节切，超 400 字按段落累积二切
    """
    chunks: list[dict] = []
    for md in sorted(KB_DIR.glob("*.md")):
        lines = md.read_text(encoding="utf-8").splitlines()
        # 从尾部剥离来源脚注（连续的空行 / "> " 引用行）
        while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith("> ")):
            lines.pop()
        body = "\n".join(lines).strip()
        title_line = next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), md.stem)

        qa_idx = 0
        para_buf: list[str] = []

        def flush_para() -> None:
            """把累积的普通段落按旧逻辑切片入库"""
            nonlocal para_buf, qa_idx
            text = "\n".join(para_buf).strip()
            para_buf = []
            if not text:
                return
            for i, sec in enumerate(s for s in re.split(r"\n(?=## )", text) if s.strip()):
                sec = sec.strip()
                pieces = [sec] if len(sec) <= CHUNK_MAX else _split_by_paragraph(sec)
                for j, piece in enumerate(pieces):
                    chunks.append({
                        "id": hashlib.sha1(f"{md.name}::p{qa_idx}_{j}".encode()).hexdigest()[:16],
                        "text": piece,
                        "meta": {"source": md.name, "title": md.stem, "chunk": qa_idx + j},
                    })

        for line in body.splitlines():
            s = line.strip()
            # 问答对照行：单条成 chunk（带文件标题，保持自包含）
            if s.startswith("- 「") and "→" in s:
                flush_para()
                chunks.append({
                    "id": hashlib.sha1(f"{md.name}::qa{qa_idx}".encode()).hexdigest()[:16],
                    "text": f"{title_line}｜{s}",
                    "meta": {"source": md.name, "title": md.stem, "chunk": f"qa{qa_idx}"},
                })
                qa_idx += 1
            else:
                para_buf.append(line)
        flush_para()
    return chunks


def _split_by_paragraph(sec: str) -> list[str]:
    """超长小节：按空行分段累积，到上限即截断成新片"""
    out, buf = [], ""
    for para in sec.split("\n\n"):
        if buf and len(buf) + len(para) + 2 > CHUNK_MAX:
            out.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        out.append(buf.strip())
    return out


def embed(texts: list[str]) -> list[list[float]]:
    """分批调用 SiliconFlow embedding API，返回与输入同序的向量列表"""
    vecs: list[list[float]] = []
    with httpx.Client(timeout=60) as client:
        for i in range(0, len(texts), BATCH):
            resp = client.post(
                EMBED_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": MODEL, "input": texts[i:i + BATCH]},
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])  # 按 index 对齐顺序
            vecs.extend(d["embedding"] for d in data)
            print(f"      embedding {min(i + BATCH, len(texts))}/{len(texts)}")
    return vecs


def main() -> None:
    if not API_KEY:
        sys.exit("[x] 缺 SILICONFLOW_API_KEY：请复制 .env.example 为 .env 并填入（注册 https://cloud.siliconflow.cn）")
    if not KB_DIR.exists():
        sys.exit(f"[x] 知识源目录不存在：{KB_DIR}")

    chunks = load_chunks()
    print(f"[1/3] 从 knowledge/ 读到 {len(chunks)} 个 chunk（上限 {CHUNK_MAX} 字）")
    for c in chunks:
        print(f"      - {c['meta']['source']}#chunk{c['meta']['chunk']}  {len(c['text'])} 字")

    print(f"[2/3] 调 embedding API（{MODEL}）...")
    vectors = embed([c["text"] for c in chunks])
    print(f"      完成，向量维度 {len(vectors[0])}")

    print("[3/3] 写入 Chroma ...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # 关键：bge 用 cosine，Chroma 默认 l2
    )
    # 先按来源清旧 chunk：知识卡改动后 chunk 数可能变少，upsert 只覆盖同 id，
    # 多出来的旧 chunk 会残留（chunk 漂移）→ 变更新前先整文件删除
    for source in {c["meta"]["source"] for c in chunks}:
        col.delete(where={"source": source})
    col.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[c["meta"] for c in chunks],
    )
    print(f"      collection '{COLLECTION_NAME}' 现有 {col.count()} 条向量（持久化于 {CHROMA_DIR}）")
    print('\n[ok] 入库完成。检索验证：python rag/retrieve.py "问题"')


if __name__ == "__main__":
    main()
