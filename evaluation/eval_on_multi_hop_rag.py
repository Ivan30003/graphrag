import json
from tqdm import tqdm
import re
from collections import Counter
import argparse

# Read files
with open('/home/person/projects/qa_with_kg/experiments/full_experiments/multiHopRag_q100_qwen_4b_copy/converted_qwen4b_qwen4b_predictions.json', 'r') as file:
    doc_data = json.load(file)

with open('/home/person/projects/qa_with_kg/datasets/converted/MultiHopRAG_questions_100.json', 'r') as file:
    query_data = json.load(file)

# Initialize dictionary to save lists of predictions and gold standards for each question_type
type_data = {}
overall_pred_list = []
overall_gold_list = []

# Function to get the correct answer
def get_gold(query):
    for q in query_data:
        if q['query'] == query:
            return q['answer']
    return ''

# Function to check if there is an intersection of words between two strings
def has_intersection(a, b):
    a_words = set(a.split())
    b_words = set(b.split())
    return len(a_words.intersection(b_words)) > 0

# Function to extract the answer
def extract_answer(input_string):
    match = re.search(r'The answer to the question is "(.*?)"', input_string)
    return match.group(1) if match else input_string


# Function to calculate evaluation metrics
def calculate_metrics(pred_list, gold_list):
    print(f"{len(gold_list)} | {len(pred_list)}")
    for gt, pred in zip(gold_list, pred_list):
        print(f"GT: {gt}")
        print(f"PR: {pred}")
        tp = True if has_intersection(pred.lower(), gt.lower()) else False
        fp = True if not has_intersection(pred.lower(), gt.lower()) else False
        if tp:
            print(f"TRUE-POSITIVE")
        elif fp:
            print(f"FALSE-POSITIVE")
        else:
            raise ValueError()
        print(f'------------------------')

    tp = sum(1 for pred, gold in zip(pred_list, gold_list) if has_intersection(pred.lower(), gold.lower()))
    fp = sum(1 for pred, gold in zip(pred_list, gold_list) if not has_intersection(pred.lower(), gold.lower()))
    fn = len(gold_list) - tp
    tn = len(pred_list) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return precision, recall, f1, accuracy


def main():
    # Main loop, iterate through document data
    for d in tqdm(doc_data):
        model_answer = d['model_answer']
        if 'The answer' in model_answer:
            model_answer = extract_answer(model_answer)
        gold = get_gold(d['query'])
        if gold:
            question_type = d['question_type']
            if question_type not in type_data:
                type_data[question_type] = {'pred_list': [], 'gold_list': []}
            type_data[question_type]['pred_list'].append(model_answer)
            type_data[question_type]['gold_list'].append(gold)
            overall_pred_list.append(model_answer)
            overall_gold_list.append(gold)

    # Output evaluation data for each question_type
    for question_type, data in type_data.items():
        precision, recall, f1, accuracy = calculate_metrics(data['pred_list'], data['gold_list'])
        print(f"Question Type: {question_type}")
        print(f" Precision: {precision:.2f}")
        print(f" Recall: {recall:.2f}")
        print(f" F1 Score: {f1:.2f}")
        print(f" accuracy: {accuracy:.2f}")

    # Calculate overall evaluation metrics
    overall_precision, overall_recall, overall_f1, overall_accuracy = calculate_metrics(overall_pred_list, overall_gold_list)
    print(f"Overall Metrics:")
    print(f" Precision: {overall_precision:.2f}")
    print(f" Recall: {overall_recall:.2f}")
    print(f" F1 Score: {overall_f1:.2f}")
    print(f" Accuracy: {overall_accuracy:.2f}")


if __name__ == '__main__':
    main()
