import json
import random


random.seed(42)

datasetname = ""
intput_filename = f"{datasetname}.jsonl"
test_samples = []
with open(intput_filename, "r") as f:
    for line in f:
        item = json.loads(line)
        # breakpoint()
        if item["topic"] != "product_review":
            continue
        history = item["behaviors"]
        for his in history:
            if his["rejected"] is None:
                his["rejected"] = "No negative sample."
        last = {
            "query": item["input"],
            "answer": item["output"],
        }
        test_samples.append(
            {
                "userid": item["uid"],
                "history": history,
                "target": last,
            }
        )

num = min(1000, len(test_samples))
final_samples = random.sample(test_samples, num)
output_filename = f"product_review_upi.jsonl"
with open(output_filename, "w") as f:
    for item in final_samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
