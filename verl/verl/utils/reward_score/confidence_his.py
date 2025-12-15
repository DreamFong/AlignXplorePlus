# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import math
import random

from verl.utils.reward_score.math import is_equiv
from openai import OpenAI


urls = {
    "Qwen3-8B": "http://33.184.124.242:8000/v1",
    "Qwen3-8B-h2018": "http://33.184.125.52:8000/v1"
}
client = OpenAI(
    base_url=urls["Qwen3-8B-h2018"],
    api_key="EMPTY",
)
with open(
    "/ossfs/workspace/nas/yuting/code/PreInf/reinforcement learning/playground/preference.txt",
    "r",
    encoding="utf-8",
) as f:
    system_prompt = f.read()


def extract_preference(response):
    import re

    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = re.findall(pattern, response)
    if matches:
        result = matches[-1].replace("answer here", "")
    else:
        result = ""

    return result


def _reward_template(post, preference):
    
    chosen = post["chosen"]
    rejected = post["rejected"]
    task = post["post"]
    flag = random.randint(0, 1)
    if flag:
        responseA = chosen
        responseB = rejected
        answer = "Response A"
    else:
        responseA = rejected
        responseB = chosen
        answer = "Response B"

    prop = f'Determine which response the user prefers based on the user’s preferences. Please output your selection below in a json format by filling in the placeholders in []:{{"selection": "[Response A / Response B]"}}\n{system_prompt}\n<Prompt>\n{task}\n</Prompt>\n\n<Preference>\n{preference}</Preference>\n\n<Response A>\n{responseA}\n</Response A>\n\n<Response B>\n{responseB}\n</Response B>\n\n'
    messages = [
        {"role": "system", "content": "Generate a task-specific response."},
        {"role": "user", "content": prop},
    ]

    return messages, answer


def generate_final_answer(messages):
    models = client.models.list()
    model = models.data[0].id
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # max_tokens=2048,
        stream=False,  # 在此模板中，我们等待完整响应
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        logprobs=True,
    )
    return response.choices[0].message.content, response.choices[0].logprobs.content


def extract_confidence(logprobs_content, answer):
    def normalize_token(tok: str) -> str:
        return tok.replace("Ġ", " ").replace("Ċ", "\n")
    confidence_of_A = None
    confidence_of_B = None
    # 打印所有 token 的信息，便于调试
    # print("\n--- 所有生成Token的详细信息 ---")
    # for item in logprobs_content:
    #     print(f"Token: '{item.token}', LogProb: {item.logprob:.4f}")
    #     breakpoint()

    for item in logprobs_content:
        # 注意：Token可能包含前导空格，例如 " A"。使用 strip() 来处理这种情况。
        if normalize_token(item.token).strip() == "A":
            logprob_of_A = item.logprob
            # 第4步：转换成标准概率 (0-1)
            confidence_of_A = math.exp(logprob_of_A)
        if normalize_token(item.token).strip() == "B":
            logprob_of_B = item.logprob
            confidence_of_B = math.exp(logprob_of_B)
    if confidence_of_A is None and confidence_of_B is None:
        return -1.0
    if answer == "Response A":
        confidence_of_answer = confidence_of_A or (1 - confidence_of_B)
    elif answer == "Response B":
        confidence_of_answer = confidence_of_B or (1 - confidence_of_A)
    return confidence_of_answer


def compute_score(solution_str, ground_truth, data_source, extra_info):
    """The scoring function for AlingXplore.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    posts = extra_info["posts"][:-1]
    score = 0.0
    preference = extract_preference(solution_str)

    n = 1
    posts = random.sample(posts, n)

    for post in posts:
        reward_prompt, answer = _reward_template(post, preference)
        reward_response, logprobs = generate_final_answer(reward_prompt)
        if reward_response is None:
            raise ValueError("======== API receives none content. ========")
        reward = extract_confidence(logprobs, answer)
        score += reward
    return score / n

# def compute_score(solution_str, ground_truth, data_source, extra_info):
#     """The scoring function for AlingXplore.

#     Args:
#         solution_str: the solution text
#         ground_truth: the ground truth
#     """
#     posts = extra_info["posts"]
#     score = 0.0
#     preference = extract_preference_list(solution_str)
#     if preference is None or len(preference) == 0:
#         print("=*" * 32)
#         print(solution_str)
#         print("*=" * 32)
#         return -2.0
#     reward_mode = "last"
#     n = 1
#     # posts = random.sample(posts, n)
#     if reward_mode == "last":
#         posts = posts[-n:]
#     elif reward_mode == "random":
#         posts = random.sample(posts, n)
#     elif reward_mode == "all":
#         pass
#     else:
#         raise NotImplementedError
#     for post in posts:
#         rating_prompt, flag = _reward_template(post, preference)
#         rating_response = generate(rating_prompt)
#         if rating_response is None:
#             raise ValueError("======== API receives none content. ========")
#         rating_delta = extract_ratings(rating_response)
#         if rating_delta is None:
#             warnings.simplefilter("always")
#             warnings.warn("======== Rating is none. ========")
#         else:
#             score += flag * rating_delta.sum() / len(rating_delta)
#     return score / n