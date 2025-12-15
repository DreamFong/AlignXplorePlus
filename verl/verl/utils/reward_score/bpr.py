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
import ast
import json
import warnings
import random

from openai import OpenAI
import numpy as np


urls = {
    "Qwen3-8B-H20-server-1": "http://33.184.123.246:8000/v1",
    "Qwen3-8B-H209": "http://33.184.124.128:8000/v1",
    "Qwen3-8B-h2011": "http://33.184.124.189:8000/v1",
}
client = OpenAI(
    base_url=urls["Qwen3-8B-h2011"],
    api_key="EMPTY",
)
with open(
    "/ossfs/workspace/nas/yuting/code/PreInf/reinforcement learning/playground/rating_list.txt",
    "r",
    encoding="utf-8",
) as f:
    system_prompt = f.read()


def extract_preference_list(response):
    import re

    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = re.findall(pattern, response)
    if matches:
        result = matches[-1].replace("answer here", "")
    else:
        result = ""

    try:
        ret = eval(result)
        return ret
    except:
        if "," in result:
            return [item.strip() for item in result.split(",") if item.strip()]
        elif ";" in result:
            return [item.strip() for item in result.split(";") if item.strip()]
        else:
            return [result.strip()]


def _reward_template(post, preference):

    chosen = post["chosen"]
    rejected = post["rejected"]
    task = post["post"]
    flag = random.randint(0, 1)
    if flag:
        responseA = chosen
        responseB = rejected
        answer = 1.0
    else:
        responseA = rejected
        responseB = chosen
        answer = -1.0
    prop = f"""You are an AI evaluator. Analyze the given response and score its alignment with the user's preference.\n\nFor each dimension in the preference, assign a score from -1.0 to 1.0:\n*   **-1.0:** The response clearly exhibits a characteristic opposite to the described preference.\n*   **0.0:** The response is neutral regarding the preference, or the preference is not applicable/observable.\n*   **1.0:** The response perfectly aligns with or exemplifies the described preference.\n*   Use intermediate values (e.g., 0.5, -0.7) for partial alignment/misalignment.\n\nPlease output your rating below in a json format by filling in the placeholders in []:{{"Score A": "[]", "Score B": "[]"}} with the scores in the exact same order as the preference dimensions. \n\n{system_prompt}\n\n<Prompt>\n{task}\n</Prompt>\n\n<Preference>\n{preference}\n</Preference>\n\n<Response A>\n{responseA}\n</Response A>\n\n<Response B>\n{responseB}\n</Response B>\n"""

    # prop = f"""You are an AI evaluator. Your task is to compare two responses, A and B, and score their alignment with each dimension in the provided preference list.\n\n**Scoring Instructions:**\nFor each preference dimension, assign a score from -1.0 to 1.0 to **both** Response A and Response B.\n*   **1.0:** Perfect alignment with the preference.\n*   **-1.0:** Direct opposition to the preference.\n*   **0.0:** Neutral, or the preference is not applicable.\n*   Use intermediate values (e.g., 0.5, -0.7) for partial alignment.\n\n**Output Format Instructions:**\nYour output must be a single JSON object.\n*   The **keys** of the object must be the **exact dimension names** from the preference list.\n*   The **value** for each dimension key must be another JSON object with two keys: `"score A"` and `"score B"`.\n\n--- Example ---\n{system_prompt}\n--- End of Example ---\n\n<Prompt>\n{task}\n</Prompt>\n\n<Preference>\n{preference}\n</Preference>\n\n<Response A>\n{responseA}\n</Response A>\n\n<Response B>\n{responseB}\n</Response B>\n"""

    # prop = f"""You are an AI evaluator. Your task is to compare two responses, Response A and Response B, against a given list of Preference Dimensions. For each dimension, you must determine which response the user prefers according to the preference. For i-th dimension in the preference list: If Response A is better, the i-th element of the output list must be 1. If Response B is better, the i-th element of the output list must be -1. Your output must be a JSON object by filling in the placeholder in []:{{"selection": "[]"}}. The length of the list must exactly match the number of Preference Dimensions.\n\n--- Example ---\n{system_prompt}\n--- End of Example ---\n\n<Prompt>\n{task}\n</Prompt>\n\n<Preference>\n{preference}\n</Preference>\n\n<Response A>\n{responseA}\n</Response A>\n\n<Response B>\n{responseB}\n</Response B>\n"""

    messages = [
        {"role": "system", "content": "Generate a task-specific response."},
        {"role": "user", "content": prop},
    ]

    return messages, answer


def generate(messages):
    models = client.models.list()
    model = models.data[0].id
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # temperature=1.0,
        # max_completion_tokens=2048,
        stream=False,  # 在此模板中，我们等待完整响应
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content


# def extract_last_json(llm_output: str) -> dict | None:
#     """
#     从字符串中提取最后一个完整的JSON对象并解析。此方法从后向前扫描，以确保提取的是最终结果，即使推理过程中也包含了其他JSON对象。

#     Args:
#         llm_output: 大语言模型的原始字符串输出。

#     Returns:
#         一个Python字典，如果成功解析。
#         None，如果没有找到有效的JSON对象。
#     """
#     # 1. 从字符串末尾找到最后一个右大括号 '}'
#     try:
#         end_index = llm_output.rindex("}")
#     except ValueError:
#         # 如果连 '}' 都没有，肯定没有JSON
#         print("错误：字符串中未找到 '}'。")
#         return None

#     # 2. 从最后一个 '}' 的位置向前寻找匹配的左大括号 '{'
#     # 使用一个计数器来处理嵌套的括号
#     balance = 0
#     start_index = -1

#     # 从 end_index 开始向前遍历
#     for i in range(end_index, -1, -1):
#         char = llm_output[i]
#         if char == "}":
#             balance += 1
#         elif char == "{":
#             balance -= 1

#         # 当 balance 变为 0 时，我们找到了与之匹配的 '{'
#         if balance == 0:
#             start_index = i
#             break

#     # 3. 如果没有找到匹配的 '{' (start_index 仍然是 -1)，说明格式不正确
#     if start_index == -1:
#         print("错误：找到了 '}' 但没有找到与之匹配的 '{'。")
#         return None

#     # 4. 提取并尝试解析JSON
#     json_candidate = llm_output[start_index : end_index + 1]

#     # 5. 清洗常见的LLM生成的JSON错误
#     # 修复错误：将无效的 \' 转义替换为有效的 '
#     # 在Python字符串中，'\\' 表示一个实际的反斜杠
#     cleaned_candidate = json_candidate.replace("\\'", "'")

#     try:
#         return json.loads(cleaned_candidate)
#     except json.JSONDecodeError as e:
#         print(f"错误：提取的字符串不是一个有效的JSON。错误信息: {e}")
#         print(f"待解析内容：\n---\n{cleaned_candidate}\n---")
#         return None


# def extract_ratings(llm_output: str):
#     """
#     Extracts a JSON-like object from an LLM's text output, corrects common
#     formatting errors (like single quotes or trailing commas), and converts
#     the scores into two separate lists for 'score A' and 'score B'.

#     Args:
#         llm_output: The raw string output from the language model.

#     Returns:
#         A tuple containing two lists of floats: (scores_for_A, scores_for_B).
#         Returns (None, None) if the object cannot be found or parsed.
#     """
#     # Once we have the data (as a Python dict), extract the scores.
#     final_json = extract_last_json(llm_output)
#     if final_json is None:
#         return None
#     try:
#         scores_A = []
#         scores_B = []
#         for dimension_scores in final_json.values():
#             if not isinstance(dimension_scores, dict):
#                 raise ValueError(
#                     f"Expected a dictionary for dimension scores, but got {type(dimension_scores)}"
#                 )

#             score_a = dimension_scores.get("score A")
#             score_b = dimension_scores.get("score B")

#             # Check if keys exist
#             if score_a is None:
#                 # Try to find case-insensitive 'score a'
#                 score_a = dimension_scores.get("score a")
#             if score_b is None:
#                 score_b = dimension_scores.get("score b")

#             if score_a is None or score_b is None:
#                 print(f"Missing 'score A' or 'score B' in dimension: {final_json}")
#                 return None

#             scores_A.append(float(score_a))
#             scores_B.append(float(score_b))
#         ret = np.array(scores_A, dtype="float64") - np.array(scores_B, dtype="float64")
#         return ret
#     except (ValueError, TypeError, KeyError) as err:
#         print(f"Error: The parsed data has an unexpected format. Details: {err}")
#         return None


# def extract_ratings(response: str):
#     final_json = extract_last_json(response)
#     if final_json is None:
#         return None
#     return np.array(final_json["selection"])


def extract_ratings(response: str):
    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)

    # 查找所有匹配的JSON字符串
    all_json_strings = json_pattern.findall(response)

    if not all_json_strings:
        print("未找到任何JSON格式的字符串。")
        return None

    # 获取最后一个匹配的JSON字符串
    last_json_str = all_json_strings[-1]
    # print(f"提取到的最后一个JSON字符串（未解析）:\n{last_json_str}")

    try:
        # 尝试将字符串解析为JSON对象
        last_json_object = json.loads(last_json_str)
    except json.JSONDecodeError as e:
        print(f"错误: 最后一个匹配到的字符串不是有效的JSON。\n字符串内容:\n{last_json_str}\n错误信息: {e}")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None
    score_a = last_json_object["Score A"]
    score_b = last_json_object["Score B"]
    if type(score_a) is str:
        score_a = ast.literal_eval(score_a)
    if type(score_b) is str:
        score_b = ast.literal_eval(score_b)

    # print(f"Score A:{score_a}")
    # print(f"Score B:{score_b}")
    if type(score_a) is float:
        score_a = [score_a]
    if type(score_b) is float:
        score_b = [score_b]
    if len(score_a) != len(score_b):
        print(f"Score维度不一致{len(score_a)} vs. {len(score_b)}")
        return None
    ret = np.array(score_a, dtype="float64") - np.array(score_b, dtype="float64")
    return ret


def regularization(preferences: list, lambda1: float, lambda2: float):
    return lambda1 * len(preferences) + lambda2 * sum(len(p) for p in preferences)


def compute_score(solution_str, ground_truth, data_source, extra_info):
    """The scoring function for AlingXplore.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    posts = extra_info["posts"]
    score = 0.0
    preference = extract_preference_list(solution_str)
    if preference is None or len(preference) == 0:
        print("=*" * 32)
        print(solution_str)
        print("*=" * 32)
        return -2.0
    reward_mode = "last"
    n = 1
    # posts = random.sample(posts, n)
    if reward_mode == "last":
        posts = posts[-n:]
    elif reward_mode == "random":
        posts = random.sample(posts, n)
    elif reward_mode == "all":
        pass
    else:
        raise NotImplementedError
    for post in posts:
        rating_prompt, flag = _reward_template(post, preference)
        rating_response = generate(rating_prompt)
        if rating_response is None:
            raise ValueError("======== API receives none content. ========")
        rating_delta = extract_ratings(rating_response)
        if rating_delta is None:
            warnings.simplefilter("always")
            warnings.warn("======== Rating is none. ========")
        else:
            score += flag * rating_delta.sum() / len(rating_delta)
    return score / n

    # for _ in range(len(posts)):
    #     post = random.sample(posts, 1)[0]
    #     rating_prompt = _reward_template(post, preference)
    #     rating_response = generate(rating_prompt)
    #     print("=" * 32)
    #     print(rating_response)
    #     if rating_response is None:
    #         raise ValueError("======== API receives none content. ========")
    #     rating_delta = extract_ratings(rating_response)
    #     print(rating_delta)
    #     if rating_delta is None:
    #         warnings.simplefilter("always")
    #         warnings.warn("======== Rating is none. ========")
    #         continue
    #     else:
    #         score = rating_delta.sum() / len(rating_delta)
    #         break
    # return score

    # Vision 3
    # post = random.sample(posts, 1)[0]
    # rating_prompt = _reward_template(post, preference)
    # rating_response = generate(rating_prompt)
    # # print("=" * 32)
    # # print(rating_response)
    # if rating_response is None:
    #     raise ValueError("======== API receives none content. ========")
    # rating_delta = extract_ratings(rating_response)
    # print(rating_delta)
    # if rating_delta is None:
    #     warnings.simplefilter("always")
    #     warnings.warn("======== Rating is none. ========")
    # else:
    #     score = rating_delta.sum() / len(rating_delta)
    # return score
