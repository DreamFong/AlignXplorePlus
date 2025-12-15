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
from verl.utils.reward_score.math import is_equiv
from openai import OpenAI

urls = {
    "Qwen3-8B-H2014": "http://33.184.124.97:8000/v1"
}
client = OpenAI(
    base_url=urls["Qwen3-8B-H2014"],
    api_key="EMPTY",
)
with open(
    "/ossfs/workspace/nas/yuting/code/PreInf/reinforcement learning/playground/preference.txt",
    "r",
    encoding="utf-8",
) as f:
    system_prompt = f.read()


def _reward_template(reward_info, response_str):
    def extract_preference(response):
        import re

        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = re.findall(pattern, response)
        if matches:
            result = matches[-1].replace("answer here", "")
        else:
            result = ""

        return result

    task = reward_info["task"]
    responseA = reward_info["responseA"]
    responseB = reward_info["responseB"]
    preference = extract_preference(response_str)

    prop = f'Determine which response the user prefers based on the user’s preferences. Please output your selection below in a json format by filling in the placeholders in []:{{"selection": "[Response A / Response B]"}}\n{system_prompt}\n<Prompt>\n{task}\n</Prompt>\n\n<Preference>\n{preference}</Preference>\n\n<Response A>\n{responseA}\n</Response A>\n\n<Response B>\n{responseB}\n</Response B>\n\n'
    messages = [
        {"role": "system", "content": "Generate a task-specific response."},
        {"role": "user", "content": prop},
    ]

    return messages


def generate_final_answer(messages) -> str | None:
    models = client.models.list()
    model = models.data[0].id
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        # max_tokens=2048,
        stream=False,  # 在此模板中，我们等待完整响应
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content


def extract_final_answers(response: str) -> str:
    try:
        res = json.loads(response)
        res = res["selection"]
    except:
        try:
            match = re.search(r"\{.*\}", response)
            if match:
                json_str = match.group()
                parsed_json = json.loads(json_str)
                res = parsed_json
                res = res["selection"]
            else:
                res = response
        except:
            res = response

    return res


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None):
    """The scoring function for AlingXplore.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    reward_prompt = _reward_template(extra_info, solution_str)
    reward_response = generate_final_answer(reward_prompt)
    if reward_response is None:
        raise ValueError("======== API receives none content. ========")
    answer = extract_final_answers(response=reward_response)
    return float(is_equiv(answer, ground_truth))
