import json
import argparse

relation_set = ['per:alternate_names', 'per:alumni', 'per:place_of_residence', 'per:employee_or_member_of', 'per:girl/boyfriend', 'per:title', 'per:positive_impression', 'gpe:residents_of_place', 'org:employees_or_members', 'per:children', 'per:parents', 'per:siblings', 'per:spouse', 'per:friends', 'per:negative_impression', 'per:client', 'per:pet', 'per:place_of_work', 'per:boss', 'per:subordinate', 'per:acquaintance', 'per:roommate', 'per:dates', 'per:other_family', 'per:age', 'per:visited_place', 'gpe:visitors_of_place', 'per:origin', 'per:neighbor', 'per:works', 'per:schools_attended', 'org:students', 'per:major', 'per:date_of_birth']

parser = argparse.ArgumentParser(description="Long in-context Learning Results", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-r", "--round", type=str, help="number of rounds the context have")
parser.add_argument("-m", "--model", type=str, help="LLM model")
args = parser.parse_args()

try:
    file = open(f'./dialogueRE_round_result/{args.model}_{args.round}.json', encoding="utf8")
    data = json.load(file)
except:
    exit(1)

total_label = 0
total_pred = 0
total_correct = 0
f1 = 0.0
count = 0
for result in data:
    label = result['label'].split(",")
    label = [l.strip() for l in label]
    pred = result['pred'].split(",")
    pred = [p.strip() for p in pred]
    count += len(label)

    num_label = len(label)
    num_pred = len(pred)
    align_total = min(num_label, num_pred)
    total_label += num_label
    for pred_rela in pred:
        if any([relation in pred_rela for relation in relation_set]):
            total_pred += 1

    for idx in range(align_total):
        if label[idx] == pred[idx] or label[idx] in pred[idx]:
            total_correct += 1

    precision = total_correct / (total_pred + 1e-8)
    recall = total_correct / (total_label + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    if total_label >= 500:
        break

print('total_label: ', total_label)
print(f'f1: ', f1 * 100)