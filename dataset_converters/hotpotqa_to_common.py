import argparse
import json
from pathlib import Path
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_json_path', type=str)
    parser.add_argument('--num_questions', type=str)
    parser.add_argument('--output_folder_path', type=str)
    args = parser.parse_args()
    input_path = args.dataset_json_path
    output_folder_path = args.output_folder_path

    with open(input_path, 'r') as input_file:
        corpus = json.load(input_file)
    
    num_questions = int(args.num_questions)
    retrieval_corpus = []
    questions = []
    index = 0
    for sample in corpus:
        if len(questions) == num_questions:
            break
        question = sample['question']
        
        texts_index_list = []
        for context_text in sample['context']:
            cur_context = ' '.join(context_text[1])
            reorganized_sample = {"id": sample["_id"], "index": index, "passage": cur_context}
            retrieval_corpus.append(reorganized_sample)
            texts_index_list.append(index)
            index += 1
        question_full_info = {"id": sample["_id"], "question": question, "ref_paragraphs": texts_index_list}
        questions.append(question_full_info)
    
    print(f"{len(questions)} questions")
    print(f"{len(retrieval_corpus)} paragraphs")
    filename = Path(input_path).stem
    set_name = '_'.join(filename.split('_')[:-1])
    num_questions = 'all' if num_questions == len(corpus) else num_questions 
    output_file_path = os.path.join(Path(output_folder_path), Path(f"{set_name}_corpus_{num_questions}.json"))
    with open(output_file_path, 'w') as output_file:
        corpus = json.dump(retrieval_corpus, output_file, indent=4)
    output_file_path = os.path.join(Path(output_folder_path), Path(f"{set_name}_questions_{num_questions}.json"))
    with open(output_file_path, 'w') as output_file:
        corpus = json.dump(questions, output_file, indent=4)
