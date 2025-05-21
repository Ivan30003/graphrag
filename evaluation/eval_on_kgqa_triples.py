import argparse
import json
import csv


def simple_eval(preds, gts):
    pred_entities = []
    for sample in preds:
        for entity in sample[:-1]:
            pred_entities.append(entity.lower())

    gt_entities = []
    for sample in gts[:1]:
        all_cur_paragraphs_triples = sample['all_triples'][0]
        for triple in all_cur_paragraphs_triples:
            cur_entities = []
            for entity in list(triple.values()):
                if type(entity) == str:
                    cur_entities.append(entity.lower())
                elif type(entity) == list and len(entity) == 1:
                    cur_entities.append(entity[0].lower())
                else:
                    raise ValueError(f"{entity=}")
            gt_entities.extend(cur_entities)
            

    print(f"\n{gt_entities=}\n\n{pred_entities=}")
    unique_gts = set(gt_entities)
    unique_preds = set(pred_entities)
    intersection = list(unique_gts & unique_preds)
    print(f"\n{intersection=}")
    presicion = len(intersection) / len(pred_entities)
    recall = len(intersection) / len(gt_entities)
    print(f"\n{presicion=} | {recall=}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_path', type=str)
    parser.add_argument('--ground_truth_path', type=str)
    parser.add_argument('--mode', choices=['simple, per_paragraph'] , default='simple')
    args = parser.parse_args()
    predictions_path = args.predictions_path
    ground_truth_path = args.ground_truth_path
    # passages_num = args.num_passages
    with open(ground_truth_path, 'r') as input_file:
        ground_truth = json.load(input_file)

    if args.mode == "simple":
        predictions = []
        with open(predictions_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                for ind in range(len(row)):
                    row[ind] = row[ind].lower()
                predictions.append(row)
        simple_eval(predictions, ground_truth)