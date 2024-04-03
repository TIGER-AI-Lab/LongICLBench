import json
import re
from mosestokenizer import MosesTokenizer, MosesDetokenizer


detokenizer = MosesDetokenizer()

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def process_data_tacred(input_json_file, output_file_path):
    fileobject = open(input_json_file)
    data = json.load(fileobject)
    processed_data = []
    relation_set = []
    print(len(data))
    count = 0
    for entry in data:
        count += 1
        print(count)
        with MosesDetokenizer('en') as detokenize:
            sentence = detokenize(entry['token'])
            subject_entity = detokenize(entry['token'][entry['subj_start']:entry['subj_end']+1])
            object_entity = detokenize(entry['token'][entry['obj_start']:entry['obj_end']+1])


        relation = entry['relation']
        if relation not in relation_set:
            relation_set.append(relation)

        processed_data.append({
            "sentence": sentence,
            "relation": relation,
            "subject_entity": subject_entity,
            "object_entity": object_entity
        })
    print(relation_set)
    print(len(relation_set))

    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(processed_data, json_file, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    # preprocess for tacred
    train_json_file = 'data/tacred/data/json/train.json'
    test_json_file = 'data/tacred/data/json/test.json'
    train_output_file = 'processed_data/train_tacred.json'
    test_output_file = 'processed_data/test_tacred.json'
    process_data_tacred(train_json_file, train_output_file)
    process_data_tacred(test_json_file, test_output_file)