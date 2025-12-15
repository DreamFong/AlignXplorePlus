import json
import random

original_data = []
with open("alignx_upi.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        original_data.append(data)

noise_data = []
with open("movielens_upi.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        noise_data.append(data)


def interleave_keep_order(a, b):
    """
    a, b 为两个列表
    返回一个新列表: a 和 b 混合，且各自内部顺序不变
    """
    n, m = len(a), len(b)
    total = n + m

    # 在 0..total-1 中随机选 m 个位置给 b
    b_positions = sorted(random.sample(range(total), m))

    res = []
    i = j = 0  # i 指向 a，j 指向 b
    for pos in range(total):
        if j < m and pos == b_positions[j]:
            res.append(b[j])
            j += 1
        else:
            res.append(a[i])
            i += 1
    return res


def interleave_later(a, b):
    return a + b


def interleave_former(a, b):
    return b + a


alignx_mix_data = []
alignx_mix_data2 = []
# movie_mix_data = []
for alignx, movie in zip(original_data, noise_data):
    new_item = dict(alignx)
    noise_num = min(int(len(alignx["history"]) * 0.75), len(movie["history"]))
    noise = random.sample(movie["history"], noise_num)
    history = alignx["history"]
    new_item["history"] = interleave_keep_order(history, noise)
    alignx_mix_data.append(new_item)


with open(
    "alignx_with_movie_random_75_upi.jsonl", "w",
) as f:
    for item in alignx_mix_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

