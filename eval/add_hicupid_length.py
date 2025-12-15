import json
import random

original_data = []
with open(
    "hicupid_upi.jsonl", "r"
) as f:
    for line in f:
        data = json.loads(line)
        original_data.append(data)


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


mix_data_later = []
mix_data_former = []
mix_data_random = []

# movie_mix_data = []
for first, second in zip(original_data[0::2], original_data[1::2]):
    later = dict(first)
    noise_num = min(int(len(first["history"]) * 0.5), len(second["history"]))
    noise = random.sample(second["history"], noise_num)
    history = first["history"]
    later["history"] = interleave_later(history, noise)
    former = dict(first)
    former["history"] = interleave_former(history, noise)
    rdm = dict(first)
    rdm["history"] = interleave_keep_order(history, noise)
    mix_data_later.append(later)
    mix_data_former.append(former)
    mix_data_random.append(rdm)

with open(
    "hicupid_random_50_upi.jsonl", "w",
) as f:
    for item in mix_data_random:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
