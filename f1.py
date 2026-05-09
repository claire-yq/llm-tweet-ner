import re
import csv
from collections import defaultdict


def round_1_eval(model_name):
    for split in ["dev", "test"]:
        tp = 0
        fp = 0
        fn = 0
        wt = 0
        with open(f"115-data/{model_name}-{split}-true.txt", "r", encoding="utf-8") as file_true, \
                open(f"115-data/{model_name}-{split}-pred.txt", "r", encoding="utf-8") as file_pred:
            for line_num, (true, pred) in enumerate(zip(file_true, file_pred), start=1):
                true_mentions = re.findall("<<(.*?)>>", true)
                true_mentions_dict = {}
                for mention in true_mentions:
                    entity_type, mention = mention.split(": ", 1)
                    true_mentions_dict[mention] = entity_type

                pred_mentions = re.findall("<<(.*?)>>", pred)
                pred_mentions_dict = {}
                for mention in pred_mentions:
                    try:
                        entity_type, mention = mention.split(": ", 1)
                        pred_mentions_dict[mention] = entity_type
                    except:
                        print("Invalid mention: " + mention)

                for mention, true_type in true_mentions_dict.items():
                    pred_type = pred_mentions_dict.get(mention)
                    if pred_type == true_type:
                        tp += 1
                    elif pred_type is None:
                        fn += 1
                    else:  # partial credit for wrong type
                        fn += 0.5
                        fp += 0.5
                        wt += 1
                for mention in pred_mentions_dict:
                    if true_mentions_dict.get(mention) is None:
                        fp += 1

        print(f"***{model_name}-{split}***")
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print(f"{f1 * 100:.2f}")

def round_1_per_class(model_name):
    for split in ["test"]:
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        with open(f"115-data/{model_name}-{split}-true.txt", "r", encoding="utf-8") as file_true, \
                open(f"115-data/{model_name}-{split}-pred.txt", "r", encoding="utf-8") as file_pred:
            for line_num, (true, pred) in enumerate(zip(file_true, file_pred), start=1):
                true_mentions = re.findall("<<(.*?)>>", true)
                true_mentions_dict = {}
                for mention in true_mentions:
                    entity_type, mention = mention.split(": ", 1)
                    true_mentions_dict[mention] = entity_type

                pred_mentions = re.findall("<<(.*?)>>", pred)
                pred_mentions_dict = {}
                for mention in pred_mentions:
                    try:
                        entity_type, mention = mention.split(": ", 1)
                        pred_mentions_dict[mention] = entity_type
                    except:
                        print("Invalid mention: " + mention)

                for mention, true_type in true_mentions_dict.items():
                    pred_type = pred_mentions_dict.get(mention)
                    if pred_type == true_type:
                        tp[true_type] += 1
                    elif pred_type is None:
                        fn[true_type] += 1
                    else:  # partial credit for wrong type
                        fn[true_type] += 0.5
                        fp[pred_type] += 0.5
                for mention, entity_type in pred_mentions_dict.items():
                    if true_mentions_dict.get(mention) is None:
                        fp[entity_type] += 1

        print(f"***{model_name}-{split}***")
        for et in ["PER", "LOC", "ORG", "MISC"]:
            precision = tp[et] / (tp[et] + fp[et])
            recall = tp[et] / (tp[et] + fn[et])
            f1 = 2 * precision * recall / (precision + recall)
            print(f"{et}: {f1 * 100:.2f}")


def round_2_eval(model_name):
    tp = 0
    fp = 0
    fn = 0
    wt = 0
    with open(f"115-data/{model_name}-test-true.csv", "r", encoding="utf-8") as file_true, \
            open(f"115-data/{model_name}-test-pred-r2.csv", "r", encoding="utf-8") as file_pred:
        reader_true = csv.reader(file_true)
        reader_pred = csv.reader(file_pred)
        next(reader_true)  # skip header row
        next(reader_pred)
        for true, pred in zip(reader_true, reader_pred):
            if true[2] == pred[2]:
                if true[2] != "O":
                    tp += 1
            elif true[2] == "O":
                fp += 1
            elif pred[2] == "O":
                fn += 1
            else:  # partial credit for wrong type
                fn += 0.5
                fp += 0.5
                wt += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"***{model_name}***")
    print(f"{f1 * 100:.2f}")

def round_2_per_class(model_name):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    with open(f"115-data/{model_name}-test-true.csv", "r", encoding="utf-8") as file_true, \
            open(f"115-data/{model_name}-test-pred-r2.csv", "r", encoding="utf-8") as file_pred:
        reader_true = csv.reader(file_true)
        reader_pred = csv.reader(file_pred)
        next(reader_true)  # skip header row
        next(reader_pred)
        for true, pred in zip(reader_true, reader_pred):
            if true[2] == pred[2]:
                if true[2] != "O":
                    tp[true[2]] += 1
            elif true[2] == "O":
                fp[pred[2]] += 1
            elif pred[2] == "O":
                fn[true[2]] += 1
            else:  # partial credit for wrong type
                fn[true[2]] += 0.5
                fp[pred[2]] += 0.5

    print(f"***{model_name}***")
    for et in ["PER", "LOC", "ORG", "MISC"]:
        precision = tp[et] / (tp[et] + fp[et])
        recall = tp[et] / (tp[et] + fn[et])
        f1 = 2 * precision * recall / (precision + recall)
        print(f"{et}: {f1 * 100:.2f}")


if __name__ == '__main__':
    for model_name in ["llama", "mistral", "qwen"]:
        round_1_eval(model_name)
        round_1_per_class(model_name)
        round_2_eval(model_name)
        round_2_per_class(model_name)
