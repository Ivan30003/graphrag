import json
import argparse
from pathlib import Path
from sklearn.metrics import ndcg_score


def calculate_scores(ground_truth_list, predicted_list):
        return (ndcg_score(ground_truth_list, predicted_list, k=2), ndcg_score(ground_truth_list, predicted_list, k=5),
                ndcg_score(ground_truth_list, predicted_list, k=10))


def calculate_metrics(predictions_data):
    metrics = {}
    for prediction_info in predictions_data:
        ground_truth_list = prediction_info['label']
        predicted_list = prediction_info['answer']
        user_id = prediction_info['question']
        if len(predicted_list) < 10:
            zeros_num = 10 - len(predicted_list)
        score = calculate_scores([ground_truth_list], [predicted_list])
        metrics[user_id] = score

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_path', type=Path, required=True)
    # parser.add_argument('--questions_path', type=Path, required=True)

    args = parser.parse_args()
    converted_preds_path = Path(args.preds_path)
    # questions_path = Path(args.questions_path)

    # Read files
    with open(converted_preds_path, 'r') as file:
        predictions = json.load(file)

    # with open(questions_path, 'r') as file:
    #     query_data = json.load(file)

    metrics = calculate_metrics(predictions)
    
    all_ndcg2_scores = [metrics[score][0] for score in metrics]
    all_ndcg5_scores = [metrics[score][1] for score in metrics]
    all_ndcg10_scores = [metrics[score][2] for score in metrics]

    all_mean_scores = (sum(all_ndcg2_scores)/len(all_ndcg2_scores), sum(all_ndcg5_scores)/len(all_ndcg5_scores), 
                    sum(all_ndcg10_scores)/len(all_ndcg10_scores))
    
    print(all_mean_scores)

if __name__ == '__main__':
    main()
