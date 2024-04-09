from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer
from huggingface_hub import hf_hub_download
import os
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import torch
import argparse
import json
from openai import OpenAI
import anthropic
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

PROJECT_ID = "gemini-infer"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

generation_config = GenerationConfig(
    temperature=1,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=100,
)

selected_labels = ['org:founded_by', 'per:employee_of', 'org:alternate_names', 'per:cities_of_residence', 'per:children', 'per:title', 'per:siblings', 'per:religion', 'per:age', 'org:website', 'per:stateorprovinces_of_residence', 'org:member_of', 'org:top_members/employees', 'per:countries_of_residence', 'org:city_of_headquarters', 'org:members', 'org:country_of_headquarters', 'per:spouse', 'org:stateorprovince_of_headquarters', 'org:number_of_employees/members', 'org:parents', 'org:subsidiaries', 'per:origin', 'org:political/religious_affiliation', 'per:other_family', 'per:stateorprovince_of_birth', 'org:dissolved', 'per:date_of_death', 'org:shareholders', 'per:alternate_names', 'per:parents', 'per:schools_attended', 'per:cause_of_death', 'per:city_of_death', 'per:stateorprovince_of_death', 'org:founded', 'per:country_of_birth', 'per:date_of_birth', 'per:city_of_birth', 'per:charges', 'per:country_of_death']

def select_data(given_dataset, number_of_turns):
    turns = 0
    label_list = []
    selected_data_list = []
    for data in given_dataset:
        if data['relation'] not in label_list and data['relation'] in selected_labels:
            selected_data_list.append(data)
            label_list.append(data['relation'])
        if len(label_list) == len(selected_labels):
            turns += 1
            if turns == number_of_turns:
                break
            else:
                label_list = []
    return selected_data_list

def select_test(given_dataset, number_of_turns):
    selected_data_list = []
    count_dict = {rela: 0 for rela in selected_labels}
    print("==========")
    print(len(given_dataset))
    for data in given_dataset:
        if data['relation'] in selected_labels and count_dict[data['relation']] < number_of_turns:
            selected_data_list.append(data)
            count_dict[data['relation']] += 1
    return selected_data_list

def format_discovery_prompt(data_dict_list, with_instruction=False, round=0, context_token_number="2k", group=False):
    token_shot_map_dict = {"600": 5, "2k": 25, "5k": 67, "10k": 133, "15k": 204, "20k": 270, "25k": 362,
                           "32k": 421}
    prompt = 'Given a sentence and a pair of subject and object entities within the sentence, please predict the relation between the given entities.'
    if with_instruction:
        prompt = prompt + " The predicted relationship must come from these classes: "
        for i, word in enumerate(selected_labels):
            if i != len(selected_labels) - 1:
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
    print("org data_list: ", data_list)
    if group:
        print("==============demo grouped==============")
        data_list = sorted(data_list, key=lambda d: d['relation'])
        print("after grouping data_list: ", data_list)

    position_number_record = {}
    pos = 0
    for data in data_list:
        pos += 1
        if data["relation"] not in position_number_record:
            position_number_record[data["relation"]] = {}
            position_number_record[data["relation"]]["number"] = 1
            position_number_record[data["relation"]]["pos"] = [pos]
        else:
            position_number_record[data["relation"]]["number"] += 1
            position_number_record[data["relation"]]["pos"].append(pos)
    print("position_number_record: ", position_number_record)
    for data in data_list:
        prompt = prompt + "sentence: " + data['sentence'] + '\n'
        prompt = prompt + "the subject is " + data["subject_entity"] + " and the object is " + data["object_entity"] + '\n'
        prompt = prompt + "the relation between the two entities is: " + data["relation"] + '\n'
    return prompt, position_number_record

def generate_text(project_id: str, location: str, prompt: str, model) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

    # Query the model
    responses = model.generate_content(prompt,
                                       generation_config=generation_config,
                                       stream=False)
    for response in responses:
        return response.text



parser = argparse.ArgumentParser(description="Long in-context Learning",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--context_length", type=str, default='2k', help="number of tokens the context have")
parser.add_argument("-m", "--model", type=str, help="model name to test")
parser.add_argument("-g", "--group", action="store_true", help="whether to group the type of demonstration")
parser.add_argument("-k", "--api_key", type=str, help="api key of open ai")
parser.add_argument("--test_number", type=int, help="number of examples to run for test")
parser.add_argument("--round", type=int, default=0, help="number of round for demonstration")
parser.add_argument("--instruct", action="store_true", help="whether to show all the labels as instruction")
args = parser.parse_args()

test_file = open('processed_data/test_tacred.json')
test_data = json.load(test_file)
train_file = open('processed_data/train_tacred.json')
train_data = json.load(train_file)
demo_data = select_data(given_dataset=train_data, number_of_turns=args.round)
eva_data = select_test(given_dataset=test_data, number_of_turns=20)


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

demo_prompt, position_number_record = format_discovery_prompt(demo_data, with_instruction=args.instruct,
                                                              context_token_number=args.context_length,
                                                              group=args.group, round=args.round)

total_label = 0
total_pred = 0
total_correct = 0

if args.round != 0:
    if args.instruct:
        output_file = f'tacred_round_instruct_result/{args.model}_{args.round}.json'
    elif args.group:
        output_file = f'tacred_round_group_result/{args.model}_{args.round}.json'
    else:
        output_file = f'tacred_round_result/{args.model}_{args.round}.json'
else:
    if args.instruct:
        output_file = f'tacred_instruct_result/{args.model}_{args.round}.json'
    elif args.group:
        output_file = f'tacred_group_result/{args.model}_{args.context_length}.json'
    else:
        output_file = f'tacred_result/{args.model}_{args.context_length}.json'
if not os.path.exists(output_file.split('/')[0]):
    os.makedirs(output_file.split('/')[0])

with open(output_file, mode='w', encoding='utf-8') as f:
    feeds = []
    f.write(json.dumps(feeds, indent=2))

print(f"==========Evluation for {args.model}; Round {args.round}==============")
for example in eva_data[:args.test_number]:
    cur_prompt = demo_prompt + "sentence: " + example['sentence'] + '\n'
    cur_prompt = cur_prompt + "the subject is " + example["subject_entity"] + " and the object is " + example["object_entity"] + '\n'
    cur_prompt = cur_prompt + "the relation between the two entities is: "
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

    print("org response: ", response)
    temp_prompt = "the relation between the two entities is:"
    if example['sentence'] in response:
        response = list(response.split(example['sentence']))[-1].strip().split(temp_prompt)
        if len(response) > 1:
            response = response[1].split("sentence:")[0]
        else:
            response = response[0]
    else:
        response = response.split("sentence:")[0]

    response = list(response.strip().split("\n"))[0]
    label = example["relation"]
    pred = response.strip()


    print("sentence: ", example["sentence"])
    print("subject: ", example["subject_entity"])
    print("object: ", example["object_entity"])
    print("label: ", label)
    print("response: ", response)
    if pred in selected_labels:
        total_pred += 1
    if label == pred:
        total_correct += 1
    total_label += 1
    precision = total_correct / (total_pred + 1e-8)
    recall = total_correct / (total_label + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("total: ", total_label)
    print("total pred: ", total_pred)
    print("total correct: ", total_correct)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)

    output_dict = {}
    output_dict["sentence"] = example["sentence"]
    output_dict["pred"] = response
    output_dict["label"] = label
    output_dict["label_appear_num"] = position_number_record[label]["number"]
    output_dict["label_appear_pos"] = position_number_record[label]["pos"]
    feeds.append(output_dict)
    with open(output_file, mode='w', encoding='utf-8') as feedsjson:
        feedsjson.write(json.dumps(feeds, indent=2))

