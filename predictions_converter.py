import argparse
import json
# from pathlib import Path
# import os


def prepare_multi_hop_rag_preds(predictions, questions_dict):
    prepared_predictions = []
    for prediction_info in predictions:
        prepared_prediction = {}
        question_type = questions_dict[prediction_info['question']]
        prediction = prediction_info['answer']
        cleaned_prediction = ''

        if len(prediction) > 0:
            if question_type == 'inference_query':
                cleaned_prediction = prediction.strip().strip('.')
            elif question_type == 'comparison_query':  # '  No, the context says ...' -> 'No'
                cleaned_prediction = prediction.strip().replace(',', '').replace('.', '').split()[0]
            elif question_type == 'null_query':
                if "unable to determine" in prediction.lower() or "cannot determine" in prediction.lower():
                    cleaned_prediction = "Insufficient information"
                else:
                    cleaned_prediction = ''
            elif question_type == 'temporal_query':
                cleaned_prediction = prediction.strip().replace(',', '').replace('.', '').split()[0]
            else:
                raise ValueError(f"{question_type} Unknown")

        prepared_prediction['model_answer'] = cleaned_prediction
        prepared_prediction['query'] = prediction_info['question']
        prepared_prediction['question_type'] = question_type
        prepared_predictions.append(prepared_prediction)

    return prepared_predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_preds_json_path', type=str)
    parser.add_argument('--questions_json_path', type=str)
    parser.add_argument('--benchmark_name', choices=['hotPotQA', 'multiHopRAG'], type=str)
    parser.add_argument('--output_preds_path', type=str)
    args = parser.parse_args()
    input_path = args.raw_preds_json_path
    questions_json_path = args.questions_json_path
    output_path = args.output_preds_path
    benchmark_name = args.benchmark_name

    with open(input_path, 'r') as input_file:
        predictions = json.load(input_file)

    with open(questions_json_path, 'r') as input_file:
        questions = json.load(input_file)
    
    
    if benchmark_name == "multiHopRAG":
        questions_dict = {}
        for question_info in questions:
            questions_dict[question_info['query']] = question_info['question_type']
        prepared_predictions = prepare_multi_hop_rag_preds(predictions, questions_dict)
    else:
        raise NotImplementedError()
    
    # print(f"{len(questions)} questions")
    # print(f"{len(retrieval_corpus)} paragraphs")
    # filename = Path(input_path).stem
    # set_name = '_'.join(filename.split('_')[:-1])
    # num_questions = 'all' if num_questions == len(corpus) else num_questions 
    # output_file_path = os.path.join(Path(output_folder_path), Path(f"{set_name}_corpus_{num_questions}.json"))
    # with open(output_file_path, 'w') as output_file:
    #     corpus = json.dump(retrieval_corpus, output_file, indent=4)
    output_file_path = output_path
    with open(output_file_path, 'w') as output_file:
        corpus = json.dump(prepared_predictions, output_file, indent=4)
