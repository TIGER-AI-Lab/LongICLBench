import json
import argparse

parser = argparse.ArgumentParser(description="Long in-context Learning Results", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-r", "--round", type=str, help="number of rounds the context have")
parser.add_argument("-m", "--model", type=str, help="LLM model")
args = parser.parse_args()

test_file = open('processed_data/fewnerd_test_1800.json')
gt_data = json.load(test_file)

output_file = open(f'fewnerd_round_result/{args.model}_{args.round}.json')
pred_data = json.load(output_file)

all_num_example = len(pred_data)
total_pred_cnt = 0
total_label_cnt = 0
total_correct_cnt = 0

for i in range(all_num_example):
    predictions = pred_data[i]
    labels = gt_data[i]['entities']
    print("predictions: ", predictions)
    print("labels: ", labels)
    total_pred_cnt += len(predictions)
    total_label_cnt += len(labels)
    for ent in predictions:
        if ent in labels and labels[ent] == predictions[ent]:
            total_correct_cnt += 1

print("number of evaluation sentence: ", all_num_example)
print("# total_pred_cnt: ", total_pred_cnt)
print("# total_label_cnt: ", total_label_cnt)
print("# total_correct_cnt: ", total_correct_cnt)


precision = total_correct_cnt / (total_pred_cnt + 1e-8)
recall = total_correct_cnt / (total_label_cnt + 1e-8)
f1 = 2 * precision * recall / (precision + recall)


print("precision: ", precision)
print("recall: ", recall)
print("f1: ", f1)



