from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer
from huggingface_hub import hf_hub_download
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import torch
import argparse
import re
from openai import OpenAI
import anthropic
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import json
import os

PROJECT_ID = "gemini-infer"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

generation_config = GenerationConfig(
    temperature=1,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=100,
)

all_labels = ["admiration",
                "amusement",
                "anger",
                "annoyance",
                "approval",
                "caring",
                "confusion",
                "curiosity",
                "desire",
                "disappointment",
                "disapproval",
                "disgust",
                "embarrassment",
                "excitement",
                "fear",
                "gratitude",
                "grief",
                "joy",
                "love",
                "nervousness",
                "optimism",
                "pride",
                "realization",
                "relief",
                "remorse",
                "sadness",
                "surprise",
                "neutral"]


def generate_text(project_id: str, location: str, prompt: str, model) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    responses = model.generate_content(prompt,
                                       generation_config=generation_config,
                                       stream=False)
    for response in responses:
        return response.text

def select_data(given_dataset, number_of_turns):
    selected_data_list = []
    label_to_data_dict = {}
    for data in given_dataset:
        if len(data['labels']) == 1:
            cur_label = data['labels'][0]
            if cur_label in label_to_data_dict:
                label_to_data_dict[cur_label].append(data)
            else:
                label_to_data_dict[cur_label] = [data]
    data_label_list = list(label_to_data_dict.keys())
    selected_label_to_count = {key:0 for key in data_label_list}
    for turn in range(number_of_turns):
        for i, key in enumerate(data_label_list):
            if len(label_to_data_dict[key]) > selected_label_to_count[key]:
                selected_data_list.append(label_to_data_dict[key][selected_label_to_count[key]])
                selected_label_to_count[key] += 1
            else:
                for other in range(i+1, len(data_label_list)):
                    other_key = data_label_list[other]
                    if len(label_to_data_dict[other_key]) > selected_label_to_count[other_key]:
                        selected_data_list.append(label_to_data_dict[other_key][selected_label_to_count[other_key]])
                        selected_label_to_count[other_key] += 1
                        break

    print("selected_data_list: ", selected_data_list)
    print("selected data list length: ", len(selected_data_list))
    return selected_data_list

def format_discovery_prompt(data_dict_list, round=0, with_instruction=False, context_token_number="2k"):
    token_shot_map_dict = {"2k": 73, "5k": 190, "10k": 380, "15k": 560, "20k": 740, "25k": 920,
                           "32k": 1180}
    prompt = 'Given a comment, please predict the emotion category of this comment. The predict answer must come from the demonstration examples with the exact format.'
    if with_instruction:
        prompt = prompt + 'You can only make prediction from the following categories: '
        for i, word in enumerate(all_labels):
            if i != len(all_labels) - 1:
                prompt = prompt + word + ', '
            else:
                prompt = prompt + word + '.\n'
    prompt = prompt + ' The examples are as follows: \n'
    if round != 0:
        index = len(data_dict_list)
        print(f"======={round} round running========")
        print("number of instances: ", index)
    else:
        index = token_shot_map_dict[context_token_number]
    data_list = data_dict_list[:index]
    for data in data_list:
        prompt = prompt + "comment: " + data['text'] + "\nemotion category: " + all_labels[data['labels'][0]] + '\n'
    return prompt

parser = argparse.ArgumentParser(description="Long in-context Learning",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--context_length", type=str, default='2k', help="number of tokens the context have")
parser.add_argument("-m", "--model", type=str, help="model name to test")
parser.add_argument("-k", "--api_key", type=str, help="api key of open ai")
parser.add_argument("--instruct", action="store_true", help="whether to show all the labels as instruction")
parser.add_argument("--test_number", type=int, default=500, help="number of examples to run for test")
parser.add_argument("--round", type=int, default=0, help="number of round for demonstration")
args = parser.parse_args()

dataset = load_dataset("go_emotions")
train_data = dataset['train']
test_data = dataset['test']
demo_data = select_data(given_dataset=train_data, number_of_turns=args.round)
eva_data = select_data(given_dataset=test_data, number_of_turns=25)
total = 0
correct = 0


# define model path
if args.model == "glm":
    model_path = "THUDM/chatglm3-6b-32k"
elif args.model == "baichuan":
    model_path = "baichuan-inc/Baichuan2-7B-Base"
elif args.model == "llama2-7B-32K":
    model_path = 'togethercomputer/LLaMA-2-7B-32K'
elif args.model == 'yi':
    model_path = '01-ai/Yi-6B-200K'
elif args.model == "internlm":
    model_path = "internlm/internlm2-base-7b"
elif args.model == "longLora":
    model_path = 'Yukang/Llama-2-7b-longlora-100k-ft'
elif args.model == "longllama":
    model_path = 'syzymon/long_llama_code_7b'
elif args.model == "qwen":
    model_path = 'Qwen/Qwen1.5-7B'
elif args.model == "mistral":
    model_path = 'TIGER-Lab/Mistral-7B-Base-V0.2'
elif args.model == "gemma":
    model_path = "google/gemma-7b"
elif args.model == 'rwkv':
    title = "RWKV-5-World-7B-v2-20240128-ctx4096"
    model_path = hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename=f"{title}.pth")
elif args.model == 'gpt4':
    model_path = 'gpt-4-turbo-preview'
elif args.model == 'claude3':
    model_path = "claude-3-opus-20240229"
elif args.model == 'mamba':
    model_path = 'state-spaces/mamba-2.8b'
elif args.model == 'gemini':
    model_path = "gemini-1.0-pro"

# load tokenizer
if args.model == 'yi':
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
elif args.model == 'rwkv':
    model = RWKV(model=model_path, strategy='cuda fp16i8 *8 -> cuda fp16').cuda()
    tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
elif args.model == 'gpt4':
    model = OpenAI(api_key=args.api_key)
    tokenizer = None
elif args.model == 'gemini':
    model = GenerativeModel(model_path)
    tokenizer = None
elif args.model == 'claude3':
    model = anthropic.Anthropic(
        api_key=args.api_key,
    )
    tokenizer = None
elif args.model == 'mamba':
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# load model
if args.model == "glm":
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).half().cuda()
elif args.model == 'rwkv' or args.model == 'gpt4' or args.model == 'claude3' or args.model == 'gemini':
    pass
elif args.model == 'internlm':
    backend_config = TurbomindEngineConfig(rope_scaling_factor=2.0, session_len=200000)
    model = pipeline(model_path, backend_config=backend_config)
elif args.model == 'yi' or args.model == 'qwen':
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
elif args.model == 'longllama':
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 mem_layers=[],
                                                 mem_dtype='bfloat16',
                                                 trust_remote_code=True,
                                                 mem_attention_grouping=(4, 2048),)
elif args.model == 'mamba':
    model = MambaLMHeadModel.from_pretrained(model_path, device='cuda:0', dtype=torch.float16)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda()
if args.model != 'gpt4' and args.model != 'claude3' and args.model != 'gemini' and args.model != 'internlm':
    model = model.eval()

demo_prompt = format_discovery_prompt(demo_data,
                                      round=args.round,
                                      with_instruction=args.instruct,
                                      context_token_number=args.context_length)

if args.round != 0:
    if args.instruct:
        output_file = f'goemotion_round_instruct_result/{args.model}_{args.round}.json'
    else:
        output_file = f'goemotion_round_result/{args.model}_{args.round}.json'
else:
    if args.instruct:
        output_file = f'goemotion_instruct_result/{args.model}_{args.context_length}.json'
    else:
        output_file = f'goemotion_result/{args.model}_{args.context_length}.json'
if not os.path.exists(output_file.split('/')[0]):
    os.makedirs(output_file.split('/')[0])
with open(output_file, mode='w', encoding='utf-8') as f:
    feeds = []
    f.write(json.dumps(feeds, indent=2))

print(f"==========Evluation for {args.model}; Round {args.round}==============")
for example in eva_data[:args.test_number]:
    cur_prompt = demo_prompt + "comment: " + example['text'] + "\nemotion category: "
    if args.model != 'rwkv' and args.model != 'gpt4' and args.model != 'claude3' and args.model != 'gemini':
        inputs = tokenizer(cur_prompt, return_tensors='pt')
        print(inputs['input_ids'].shape)
        if args.model == "longllama":
            inputs = inputs.input_ids
    if args.model == "glm":
        response, history = model.chat(tokenizer, cur_prompt, history=[])
    elif args.model == 'gpt4':
        input_msg = [{"role": "user", "content": cur_prompt}]
        response = model.chat.completions.create(
            model=model_path,
            messages=input_msg,
            temperature=1.0,
            max_tokens=100
        )
        response = response.choices[0].message.content
    elif args.model == 'claude3':
        response = model.messages.create(
            model=model_path,
            max_tokens=100,
            messages=[
                {"role": "user", "content": cur_prompt}
            ]
        )
        response = response.content[0].text
    elif args.model == 'gemini':
        try:
            response = generate_text(PROJECT_ID, LOCATION, cur_prompt, model)
        except:
            response = ''
    elif args.model == 'yi':
        messages = [
            {"role": "user", "content": cur_prompt}
        ]
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
                                                  return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'), max_new_tokens=100)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print("org pred: ", response)
    elif args.model == 'rwkv':
        cur_prompt = cur_prompt.strip()
        all_tokens = []
        out_last = 0
        token_count = 100
        ctx_limit = 32000
        temperature = 1
        top_p = 1
        out_str = ''
        occurrence = {}
        state = None

        for i in range(int(token_count)):
            out, state = model.forward(tokenizer.encode(cur_prompt )[-ctx_limit:] if i == 0 else [token], state)
            for n in occurrence:
                out[n] -= (0.1 + occurrence[n] * 0.1)
            token = tokenizer.sample_logits(out, temperature=temperature, top_p=top_p)
            if token in [0]:
                break
            all_tokens += [token]
            for xxx in occurrence:
                occurrence[xxx] *= 0.996
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp:
                out_str += tmp
                out_last = i + 1
        response = out_str
    elif args.model == 'longllama':
        inputs = inputs.to('cuda:0')
        response = model.generate(
            input_ids=inputs,
            max_new_tokens=100,
            num_beams=1,
            last_context_length=3996,
            do_sample=True,
            temperature=1.0,
            eos_token_id=tokenizer.encode('sentence')
        )
        response = tokenizer.decode(response[0], skip_special_tokens=True)
    elif args.model == 'internlm':
        response = model(cur_prompt)
        response = response.text
    elif args.model == 'mamba':
        inputs = inputs.to('cuda:0')
        attn_mask = inputs.attention_mask.to(device='cuda:0')
        input_ids = inputs.input_ids.to(device='cuda:0')

        max_length = input_ids.shape[1] + 100
        print("max_length: ", max_length)

        fn = lambda: model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=1,
            top_k=1,
            top_p=1,
            repetition_penalty=0,
        )

        out = fn()
        response = tokenizer.batch_decode(out.sequences.tolist())[0]
    else:
        inputs = inputs.to('cuda:0')
        response = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(response.cpu()[0], skip_special_tokens=True)

    print("org response: ", response)
    temp_prompt = "emotion category:"
    if example['text'] in response:
        response = list(response.split(example['text']))[-1].strip().split(temp_prompt)
        if len(response) > 1:
            response = response[1].split("comment: ")[0]
        else:
            response = response[0]
    else:
        response = response.split("comment: ")[0]
    response = response.strip().split("\n")[0]

    response = response.lower().strip()
    print("pred: ", response)
    label = all_labels[example['labels'][0]]
    label = label.lower()
    print("label: ", label)
    if response == label:
        correct += 1
    total += 1
    print("accuracy: ", correct/total)
    print("correct: ", correct)
    print("all: ", total)

    output_dict = {}
    output_dict['text'] = example['text']
    output_dict['label'] = label
    output_dict['pred'] = response
    feeds.append(output_dict)
    with open(output_file, mode='w', encoding='utf-8') as feedsjson:
        feedsjson.write(json.dumps(feeds, indent=2))

