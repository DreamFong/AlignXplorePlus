# -*- coding: utf-8 -*-
"""
Step 1 (revised):
- 仅读取 dialogue:test
- 读取 qa:test_2 并筛出 type == 'persona'
- 按每条 QA 的 dialogue_id 在 dialogue:test 中收集对应对话的全部轮次并拼接 transcript
"""
import random
import json
import os

from collections import defaultdict
from datasets import load_dataset


# -----------------------
# 0) 基本配置
# -----------------------
DATASET = "benchmarks/HiCUPID/data"
CFG_QA = "qa"
CFG_DIALOGUE = "dialogue"
SPLIT_QA = "test_2"
SPLIT_DIALOGUE = "test"  # 只读 test

# -----------------------
# 1) 加载数据
# -----------------------
qa_ds = load_dataset(DATASET, CFG_QA, split=SPLIT_QA)
dlg_ds = load_dataset(DATASET, CFG_DIALOGUE, split=SPLIT_DIALOGUE)

print(f"[info] qa:test_2 size = {len(qa_ds)}")
print(f"[info] dialogue:test size = {len(dlg_ds)}")
print("[info] qa sample keys:", list(qa_ds.features.keys()))
print("[info] dialogue sample keys:", list(dlg_ds.features.keys()))

# -----------------------
# 2) 仅保留 persona 类型的 QA
# -----------------------
qa_persona = qa_ds.filter(lambda ex: ex["type"] == "persona")
print(f"[info] qa_persona count = {len(qa_persona)}")

# -----------------------
# 3) 按 dialogue_id 分组 dialogue:test，并按 turn_id 排序
# -----------------------
dialogues = defaultdict(list)
for row in dlg_ds:
    dialogues[row["dialogue_id"]].append(row)

for did, rows in dialogues.items():
    if rows and "turn_id" in rows[0]:
        rows.sort(key=lambda r: r["turn_id"])


def rows_to_transcript(rows):
    """将该 dialogue 的所有轮次拼接为文本。"""
    conversations = []
    for r in rows:
        u = r.get("user") or r.get("human") or r.get("query") or ""
        a = r.get("assistant") or r.get("bot") or r.get("response") or ""
        conversations.append(
            {
                "role": "user",
                "content": u,
            }
        )
        conversations.append(
            {
                "role": "assistant",
                "content": a,
            }
        )
    return conversations 


# -----------------------
# 4) 为每条 persona QA 找到 dialogue:test 中对应对话并拼接 transcript
# -----------------------
records = []
missing = 0
for i, rec in enumerate(qa_persona):
    # 兼容：可能是 dialogue_id（单个）或 dialogue_ids（列表）
    if "dialogue_id" in rec and rec["dialogue_id"] is not None:
        dids = rec["dialogue_id"]
    else:
        dids = []
        missing += 1
        continue
    history = []
    for did in dids:
        rows = dialogues.get(did, [])
        if not rows:
            missing += 1
            continue
        transcript = rows_to_transcript(rows)
        history += transcript
    persona = rec.get("metadata")["persona"]
    records.append(
        {
            "qa_index": i,
            "dialogue_id": dids,
            "type": rec["type"],
            "target": {
                "query": rec.get("question", ""),
                "chosen": rec.get("personalized_answer", ""),
                "rejected": rec.get("general_answer", ""),
            },
            "history": history,
            "golden": f'The user {persona["relation"]} {persona["entity"]}.',
        }
    )

print(f"[info] built records = {len(records)}, missing = {missing}")
if records:
    print("[info] example record keys:", list(records[0].keys()))
with open("upi_benchmark/hicupid_upi.jsonl", "w", encoding="utf-8") as f:
    for ex in records:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
