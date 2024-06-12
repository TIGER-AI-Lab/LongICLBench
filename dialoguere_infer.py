from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer
from huggingface_hub import hf_hub_download
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import torch
import argparse
import json
import anthropic
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import tiktoken
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import time
from openai import OpenAI
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
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


selected_labels = ['per:alternate_names', 'per:alumni', 'per:place_of_residence', 'per:employee_or_member_of',
                   'per:girl/boyfriend', 'per:title', 'per:positive_impression', 'gpe:residents_of_place',
                   'org:employees_or_members', 'per:children', 'per:parents', 'per:siblings', 'per:spouse',
                   'per:friends', 'per:negative_impression', 'per:client', 'per:pet', 'per:place_of_work', 'per:boss',
                   'per:subordinate', 'per:acquaintance', 'per:roommate', 'per:dates', 'per:other_family', 'per:age',
                   'per:visited_place', 'gpe:visitors_of_place', 'per:origin', 'per:neighbor', 'per:works',
                   'per:schools_attended', 'org:students', 'per:major', 'per:date_of_birth']

def generate_text(project_id: str, location: str, prompt: str, model) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

    # Query the model
    responses = model.generate_content(prompt,
                                       generation_config=generation_config,
                                       stream=False)
    return responses.text

def merge_data(data):
    dialogue_map = {}
    for d in data:
        if d["dialogue"] not in dialogue_map:
            dialogue_map[d["dialogue"]] = []
        dialogue_map[d["dialogue"]].append({
            "relation": d["relation"],
            "subject_entity": d["subject_entity"],
            "object_entity": d["object_entity"]
        })
    processed_data = []
    for dialogue, relations in dialogue_map.items():
        processed_data.append({
            "dialogue": dialogue,
            "relations": relations
        })
    return processed_data


def format_long_prompt(train_json_file, with_instruction=False, context_round=1):
    token_shot_map_dict = {"1": 34, "2": 67, "3": 99, "4": 131, "5": 163}
    f = open(train_json_file, encoding="utf8")
    train_data = json.load(f)
    if with_instruction:
        prompt_inst = "Given the dialogue, please find the name pair entities in the dialogue and their corresponding relation types in the strict format of given examples above, please only strictly choose from the following relation types (note that the number of entities has to strictly have the same value as the number of respective relation): "
        for i in range(len(selected_labels)):
            if i != len(selected_labels) - 1:
                prompt_inst = prompt_inst + selected_labels[i] + ", "
            else:
                prompt_inst = prompt_inst + selected_labels[i] + "." + "\n"
    else:
        prompt_inst = "Given the dialogue, please find the name pair entities in the dialogue and their corresponding relation types in the strict format of given examples as following (note that the number of entities has to strictly have the same value as the number of respective relation):\n"

    prompt = ''
    train_data = train_data[:token_shot_map_dict[context_round]]
    train_data = merge_data(train_data)
    for data in train_data:
        prompt = prompt + "Dialogue: " + '\n' + data['dialogue'] + '\n'

        entity_pairs = []
        relations = []
        for rel in data['relations']:
            entity_pairs.append(str((rel["subject_entity"], rel["object_entity"])))
            relations.append(rel["relation"])

        prompt = prompt + "The list of " + str(len(entity_pairs)) + " entity pairs are " + ", ".join(entity_pairs) + "\n"
        prompt = prompt + "The " + str(
            len(relations)) + " respective relations between each entity pair are: " + ", ".join(relations) + "\n"
    prompt = prompt_inst + prompt
    return prompt


parser = argparse.ArgumentParser(description="Long in-context Learning",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-r", "--round", type=str, help="number of rounds the context have")
parser.add_argument("-m", "--model", type=str, help="model name to test")
parser.add_argument("-k", "--api_key", type=str, default='', help="api key of open ai")
parser.add_argument("--instruct", action="store_true", help="whether to show all the labels as instruction")
parser.add_argument("--test_number", type=int, help="number of examples to run for test")
args = parser.parse_args()

if args.instruct:
    output_file = f'dialogueRE_round_instruct_result/{args.model}_{args.round}.json'
else:
    output_file = f'dialogueRE_round_result/{args.model}_{args.round}.json'
if not os.path.exists(output_file.split('/')[0]):
    os.makedirs(output_file.split('/')[0])

test_file = open('./processed_data/test_dialogueRE.json', encoding="utf8")
test_data = json.load(test_file)

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

demo_prompt = format_long_prompt('./processed_data/train_dialogueRE_processed.json', with_instruction=args.instruct,
                                 context_round=args.round)

try:
    result_file = open(output_file)
    result_data = json.load(result_file)
except:
    result_data = []

result_data_size = 0
for instance in result_data:
    result_data_size += len(instance["label"].split(", "))

if result_data_size >= args.test_number:
    exit(0)

with open(output_file, mode='w', encoding='utf-8') as f:
    feeds = []
    feeds = feeds + result_data
    f.write(json.dumps(feeds, indent=2))

total_label = 0
total_pred = 0
total_correct = 0
count = 0

for example in test_data[:args.test_number]:
    cur_prompt = demo_prompt + "Dialogue: " + '\n' + example['dialogue'] + '\n'
    entity_pairs = []
    expected_relations = []
    for rel in example['relations']:
        if count < result_data_size:
            count += 1
            continue
        if count >= args.test_number:
            break
        entity_pairs.append(str((rel["subject_entity"], rel["object_entity"])))
        expected_relations.append(rel["relation"])
        count += 1

    if len(entity_pairs) == 0:
        continue

    cur_prompt = cur_prompt + "The list of " + str(len(entity_pairs)) + " entity pairs are " + ", ".join(
        entity_pairs) + "\n"
    cur_prompt = cur_prompt + "The " + str(
        len(expected_relations)) + " respective relations between each entity pair are: "

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
            out, state = model.forward(tokenizer.encode(cur_prompt)[-ctx_limit:] if i == 0 else [token], state)
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

    temp_prompt = "The " + str(len(expected_relations)) + " respective relations between each entity pair are:"
    response = \
    list(response.split(example['dialogue']))[-1].strip().split(temp_prompt)[-1].strip().split('dialogue')[
        0].strip().split('Dialogue')[0].strip()
    response = response.strip().split("\n")[0]
    response = response.strip().split("[/INST]")[0]

    label = ", ".join(expected_relations)
    pred = response.strip()

    print("dialogue: \n", example["dialogue"])
    print("label: ", label)
    print("pred: ", pred)
    total_label += len(expected_relations)
    print("total: ", total_label)
    output_dict = {}
    output_dict["dialogue"] = example["dialogue"]
    output_dict["pred"] = pred
    output_dict["label"] = label
    feeds.append(output_dict)
    with open(output_file, mode='w', encoding='utf-8') as feedsjson:
        feedsjson.write(json.dumps(feeds, indent=2))
    if count >= args.test_number:
        break