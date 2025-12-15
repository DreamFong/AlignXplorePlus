import json
import random


with open("personamem.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

result = []
for uid, user in enumerate(dataset):
    negative = random.sample(user["incorrect_answers"], 1)[0]
    target = {
        "query": user["user_query"],
        "chosen": user["correct_answer"],
        "rejected": negative,
    }
    history = user["related_conversation_snippet"]
    sample = {"userid": uid, "history": history, "target": target}
    result.append(sample)

with open("personamem_upi.jsonl", "w") as fout:
    for sample in result:
        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
