# prompt_engineering.py

from modelscope import AutoModelForCausalLM, AutoTokenizer

import io
import json
import os
import sys
import time
import torch

# global configuration

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

model_list = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen1.5-0.5B-Chat"
]
MODELSCOPE_CACHE_DIR = "E:/.cache/modelscope"
TEST_CASE_FILE = "test_case.json"
OUTPUT_FILE = "modelscope_pe_output.json"

seed = 42
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}, torch random seed: {seed}")

def load_test_cases():
    """ load extern test cases: dict() """
    if not os.path.exists(TEST_CASE_FILE):
        raise FileExistsError(f"Error:File {TEST_CASE_FILE} Not found.")
    with open(TEST_CASE_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"\nLoaded test cases successfully!")
    return cases

def build_prompt(system_content: str, user_content: str, is_chat_model: bool, tokenizer):
    """
    Tool to build prompt, adapt Base / Chat models automatically
    :param system_content: system prompt
    :param user_content: user input prompt
    :param is_chat_model:
    :param tokenizer:
    :return: built prompt string
    """
    if is_chat_model:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = f"{system_content}\n{user_content}"
    return prompt

# load model
model, tokenizer = None, None
model_id = None
is_chat_model = None

for try_model_id in model_list:
    print(f"\nLoading model: {try_model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=try_model_id,
            cache_dir=MODELSCOPE_CACHE_DIR,
            local_files_only=True,  # explicitly ban access to web.
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {
            "pretrained_model_name_or_path": try_model_id,
            "cache_dir": MODELSCOPE_CACHE_DIR,
            "local_files_only": True
        }

        if device == "cuda":
            load_kwargs["dtype"] = torch.float16
        else:
            load_kwargs["dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(**load_kwargs).to(device)
        model.eval()
        model_id = try_model_id

        if "Chat" in model_id:
            is_chat_model = True
        else:
            is_chat_model = False

        print(f"Loaded model{model_id} successfully!")
        print(f"model type: {'Chat' if is_chat_model else 'Base'}")
        print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

        break

    except Exception as e:
        print(f"Failed to load {try_model_id}: {str(e)}, try to next candidate")

if model is None:
    raise RuntimeError("Failed to load all models, check net and env of modelscope.")

def test_sentiment_analysis(sentiment_texts: list, sampling_configs: dict):
    """test 1：sentiment_analysis + Comparison of multiple sample paras."""
    print("\n" + "=" * 60)
    print("test 1：sentiment_analysis + Comparison of multiple sample paras.")
    print("=" * 60)

    results = {}
    sys_prompt = "你是影评情感分析助手，请仅输出[正面,负面,中性]中的一个词。"
    for config_name, configs in sampling_configs.items():
        print(f"\n[{config_name}]参数:{configs}")
        config_res = []
        for idx, txt in enumerate(sentiment_texts):
            user_prompt = f"文本: {txt}\n 情感倾向: "
            prompt = build_prompt(sys_prompt, user_prompt, is_chat_model, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            # eval
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **configs
                )
            resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            config_res.append({"Input": txt, "Output": resp, "Configs": config_name})
            print(f"Input{idx + 1}:{txt[:45]}... → Output:{resp}")
        results[config_name] = config_res
    return results



def compare_prompt_strategy(prompt_cases: dict):
    """test 2：prompt strategy: Zero‑shot / Few‑shot / CoT"""
    print("\n" + "=" * 60)
    print("test 2：prompt strategy: Zero‑shot / Few‑shot / CoT")
    print("=" * 60)
    out_list = []
    sys_prompt = "你是文本处理助手，请严格按照提示要求输出结果。"
    for strategy_name, case_info in prompt_cases.items():
        test_text = case_info["test_text"]
        prompt_template = case_info["prompt_template"]
        user_prompt = prompt_template.format(text=test_text)

        prompt_str = build_prompt(sys_prompt, user_prompt, is_chat_model, tokenizer)
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                                    **inputs,
                                     max_new_tokens=100,
                                     do_sample=True,
                                     temperature=0.7,
                                     top_p=0.9
                                     )
        cost = time.time() - t0
        input_tok = int(inputs.input_ids.shape[1])
        output_tok = int(outputs[0].shape[0] - inputs.input_ids.shape[1])
        resp_text = tokenizer.decode(outputs[0][input_tok:], skip_special_tokens=True).strip()
        item = {
            "strategy": strategy_name,
            "model": model_id,
            "model type": "Chat" if is_chat_model else "Base",
            "input length": len(test_text),
            "prompt length": len(user_prompt),
            "input token": input_tok,
            "output token": output_tok,
            "total token": input_tok + output_tok,
            "generation time(s)": round(cost, 2),
            "generation rate(token/s)": round(output_tok / cost, 1) if cost > 0 else 0,
            "output content": resp_text
        }
        out_list.append(item)
        print(f"\nstrategy:{strategy_name}")
        print(f"generation time:{cost:.2f}s, output token:{output_tok}")
        print(f"output content: {resp_text}")
    return out_list

def test_code_gen(code_tasks: dict):
    """test 3：code generation"""
    print("\n" + "=" * 60)
    print("test 3：code generation")
    print("=" * 60)

    res_list = []
    sys_prompt = "你是Python编程助手，输出完整可运行代码。"
    for task in code_tasks:
        user_prompt = task["prompt"]
        prompt = build_prompt(sys_prompt, user_prompt, is_chat_model, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # eval
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.95
            )
        code_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        res_list.append({"task": task["name"], "prompt": task["prompt"], "code output": code_output})
        print(f"\ntask：{task['name']}")
        print(code_output)
    return res_list



def main():
    cases = load_test_cases()
    total_result = {
        "used_model": model_id,
        "model_type": "Chat" if is_chat_model else "Base",
        "sentiment_analysis": test_sentiment_analysis(cases["sentiment_test_texts"], cases["sampling_configs"]),
        "prompt_strategy": compare_prompt_strategy(cases["prompt_strategy_cases"]),
        "code_generation": test_code_gen(cases["code_gen_tasks"])
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(total_result, f, ensure_ascii=False, indent=2)
    print(f"\nAll tests end.")
    if device == "cuda":
        torch.cuda.empty_cache()
    return total_result

if __name__ == "__main__":
    main()