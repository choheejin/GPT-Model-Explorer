"""
Data Generation Process
batch_selfinstruct_generate.py

run:
# 현재 프로젝트의 generate_instruction을 실행한다.
# generate_instruction_following_data는 파라미터로 넘겨주는 값. 해당 함수를 실행하겠다~
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils

import fire


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    # 1.~ 2.~~, 3.~~ 4.Instructions 만들어줘.
    prompt += f"{idx + 2}. Instruction:"
    return prompt

# post_process_gpt3_response 함수는 GPT-3 모델의 응답을 후처리하여 추출된 지침들을 반환하는 역할을 수행합니다.
def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = ["image","images","graph","graphs","picture","pictures","file","files","map","maps","draw","plot","go to","video","audio","music","flowchart","diagram",]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})

    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    ## 내보내는 디렉터리
    output_dir="./",
    ## seed_task(초기 데이터)가 있는 경로
    seed_tasks_path="./seed_tasks.jsonl",
    ## 몇개의 지시사항을 만들지
    num_instructions_to_generate=100,
    ## 사용할 LLM 모델 이름
    model_name="text-davinci-003",
    ## 프롬프트 지시사항 개수
    num_prompt_instructions=3,
    ## 배치 사이즈: 모델 학습 중 parameter를 업데이트할 때 사용할 데이터 개수 즉, 파인튜닝에 사용할 데이터 개수: 5개
    request_batch_size=5,
    ## 모델의 출력 다양성을 조절하는 역할
    temperature=1.0,
    ## 모델이 다음 단어를 선택할 때 선택 가능한 후보 단어를 제한 -> 모델이 다음 단어를 선택할 때 전체 확률 분포에서 가장 확률이 높은 단어를 선택
    top_p=1.0,
    ## 병렬 처리를 위해 -> 프로세스의 개수를 지정.
    num_cpus=16,
):
    ## 'seed_tasks_path'에 있는 파일을 한 줄씩 반복적으로 파일의 각 줄을 JSON 형식으로 파싱한다.
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]

    ## 'seed_tasks'의 데이터들 중, instruction, input, output만 쓰겠다..
    seed_instruction_data = [
        {
            "instruction": t["instruction"],
            "input": t["instances"][0]["input"],
            "output": t["instances"][0]["output"]
        }
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    ## output 디렉터리 생성
    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0

    # load the LM-generated instructions
    machine_instruction_data = []

    ## 이미 regen.json 파일(머신, 즉 모델이 생성한 지시사항) 이 있을 경우, 해당 데이터를 불러온다.
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    """
    Rouge-L
    Rouge는 자동 요약 및 요약 평가를 위한 평가 척도로 널리 사용되는 방법론입니다.
    Rouge는 추출적 요약(extractive summarization)과 생성적 요약(abstractive summarization) 모델의 요약 결과를 비교하여
    그 품질을 평가하는 데 사용됩니다.
    use_stemmer=False는 표제어 추출(lemmatization)을 사용하지 않겠다는 의미입니다. 
    표제어 추출은 단어를 그 뿌리 형태로 변환하여 비교를 용이하게 하는 과정입니다. 
    이 코드에서는 표제어 추출을 사용하지 않고 원형 단어를 그대로 비교할 것을 나타냅니다.
    """
    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # tqdm은 Python에서 사용되는 진행 표시줄 라이브러리로, 반복문이나 작업의 진행 상황을 시각적으로 표시해줍니다.
    # tqdm.tqdm 함수를 호출하여 total 매개변수에 num_instructions_to_generate 값을 전달하여 진행 표시줄의 총 크기를 설정합니다.
    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    # Task Pool 생성
    # all_instructions -> seed_instruction_data, machine_instruction_data의 instruction
    # all_instruction_tokens -> all_instructions가 토큰화된 형태
    all_instructions = [ d["instruction"] for d in seed_instruction_data ] + \
                       [ d["instruction"] for d in machine_instruction_data ]

    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    # 100개의 instrcution을 생성한다.
    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        # request_batch_size=5,

        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            # seed_tasks의 명령어들을 랜덤으로 3개 뽑아온다.
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            # 프롬프트를 특정 형식에 맞게 인코딩 한다. 3개의 instructions 존재
            prompt = encode_prompt(prompt_instructions)
            # 배열에 저장.
            batch_inputs.append(prompt)
        # decoding_args에 디코딩에 필요한 인수들을 설정합니다.
        # 여기에는 온도(temperature), 생성할 토큰의 개수(n), 최대 토큰 길이(max_tokens), top-p 샘플링을 위한 임계값(top_p),
        # 중지 토큰(stop) 등이 포함됩니다.
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            # 이것
            stop=["\n20", "20.", "20."],
        )
        # 시간 어느정도 걸리냐 체크하기
        request_start = time.time()
        # 질문 날려요.
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        # 질문 날리고 받는데 걸리는 시간은??
        request_duration = time.time() - request_start

        # 자. 이제 시작이야~
        process_start = time.time()
        # post_process_gpt3_response 함수를 사용하여 결과를 처리하고, 새로운 지침을 instruction_data에 추가합니다.
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0

        # 새로운 지침과 기존 토큰화된 지침들과의 유사성을 계산합니다.
        # 풀(Pool)을 사용하여 병렬로 Rouge 점수를 계산합니다.
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            # 유사 점수 저장
            rouge_scores = [score.fmeasure for score in rouge_scores]
            # 가장 유사한 명령어
            # Rouge 점수를 기준으로 가장 유사한 지침들을 선택합니다. 상위 10개의 유사한 지침과 그 점수를 가진 사전을 생성합니다.
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }

            # 최대 Rouge 점수가 0.7보다 큰 경우, 해당 지침은 건너뜁니다.
            # 그렇지 않은 경우, 유사한 지침, 평균 유사성 점수, 새로운 명령어를 machine_instruction_data에 추가하고,
            # 모든 지침과 토큰화된 지침 목록에도 추가합니다. 그리고 진행 표시줄을 업데이트합니다.
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1

            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        # machine_instruction_data를 "regen.json" 파일에 저장합니다.
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
