# qwen_debug.py

# HF_ENDPOINT，avoid connection aborted.
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# set device, GPU > CPU
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# specify Hugging Face cache directory
hf_cache_dir = "E:/.cache/huggingface/hub"

# specify model id
model_id = "Qwen/Qwen3-0.6B"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_id,
    cache_dir=hf_cache_dir,
    local_files_only=True                    # explicitly ban access to web.
)

# load model and move to specific device
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    cache_dir=hf_cache_dir,
    local_files_only=True                    # explicitly ban access to web.
).to(device)

# print("Finish loading model and tokenizer!")

# input conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, introduce yourself."}
]

# tokenizer to standardize input
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# encode input
model_inputs = tokenizer([text], return_tensors="pt").to(device)

print("input text after encoded: ")
print("model_inputs.input_ids.size() = ", model_inputs.input_ids.size() )
print("model_inputs.attention_mask.size() = ", model_inputs.attention_mask.size() )
print(model_inputs)

# generate responding
# max_new_tokens denotes the most tokens does model generate.

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

print("generated_ids.size() = ", generated_ids.size())
print("generated_ids = ",generated_ids)

# remove generated token ids of input part, to only decode generated part from model

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in
    zip(model_inputs.input_ids, generated_ids)
]

print("generated_ids[0] = ", len(generated_ids[0]))
print("generated_ids[0] = ",generated_ids[0])

# decode generated token id
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\nAnswer from model:")
print(response)