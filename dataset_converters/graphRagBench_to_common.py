import argparse
import json
from pathlib import Path
import os


def prepare_corpus_dict(corpus: list):
    corpus_dict = {}
    for sample in corpus:
        title = sample['title']
        text = sample['body']
        corpus_dict[title] = text
    return corpus_dict

def resplit_text_further(text: str, symbols_limit: int):
    texts = []
    cur_text = ''
    for sub_text in text.split('.'):
        if len(cur_text) + len(sub_text) <= symbols_limit:
            cur_text += f"{sub_text}."
        else:
            texts.append(cur_text)
            cur_text = f"{sub_text}."
    
    return texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_json_path', type=Path)
    parser.add_argument('--questions_json_path', type=Path)
    parser.add_argument('--num_questions', type=str)
    parser.add_argument('--output_folder_path', type=Path)
    parser.add_argument('--symbols_limit', type=int, default=2000)
    args = parser.parse_args()
    corpus_json_path = args.corpus_json_path
    questions_json_path = args.questions_json_path
    output_folder_path = args.output_folder_path
    symbols_limit = args.symbols_limit

    with open(corpus_json_path, 'r') as input_file:
        corpus = json.load(input_file)
    with open(questions_json_path, 'r') as input_file:
        questions = json.load(input_file)
    
    corpus_dict = prepare_corpus_dict(corpus)

    if args.num_questions == 'all':
        num_questions = len(questions)
    elif args.num_questions.isdecimal:
        num_questions = int(args.num_questions)
    else:
        ValueError(f"Unknown format: {args.num_questions}")
    retrieval_corpus = []
    sampled_questions = []
    index = 0
    titles = []
    for qa_sample in questions:
        if len(sampled_questions) == num_questions:
            break
        
        for evidence in qa_sample['evidence_list']:
            if evidence["title"] in titles:
                continue
            cur_context_title = evidence['title']
            cur_text = corpus_dict[cur_context_title]
            resplitted_text = ''
            for sub_text in cur_text.split('\n\n'):
                if len(sub_text) == 0:
                    continue
                
                # check if uniting with next fragment will not exceed limit
                if len(resplitted_text) + len(sub_text) <= symbols_limit:
                    resplitted_text += f"{sub_text}\n"
                else:
                    index += 1
                    reorganized_text_sample = {"id": cur_context_title, "index": index, 
                                                "passage": resplitted_text.strip()}
                    retrieval_corpus.append(reorganized_text_sample)
                    resplitted_text = ''
                    if len(sub_text) > symbols_limit: # single text eceeds limit
                        print(f'WARNING! {len(sub_text)=} more than {symbols_limit}')
                        texts = resplit_text_further(sub_text, symbols_limit)
                        reorganized_texts_samples = [{"id": cur_context_title, "index": index+sub_index, 
                                                "passage": text} for sub_index, text in enumerate(texts)]
                        retrieval_corpus.extend(reorganized_texts_samples)
                        index += len(texts)
                    else:
                        resplitted_text = f"{sub_text}\n"
                
            if len(resplitted_text) > 0:  # take remains
                index += 1
                reorganized_text_sample = {"id": cur_context_title, "index": index, 
                                                "passage": resplitted_text.strip()}
                retrieval_corpus.append(reorganized_text_sample)

            titles.append(cur_context_title)
        sampled_questions.append(qa_sample)
    
    print(f"{len(sampled_questions)} questions")
    print(f"{len(retrieval_corpus)} paragraphs")
    filename = Path(questions_json_path).stem
    # set_name = '_'.join(filename.split('_')[:-1])
    num_questions = 'all' if num_questions == len(questions) else num_questions 
    output_file_path = os.path.join(Path(output_folder_path), Path(f"{filename}_corpus_{num_questions}.json"))
    with open(output_file_path, 'w') as output_file:
        corpus = json.dump(retrieval_corpus, output_file, indent=4)
    output_file_path = os.path.join(Path(output_folder_path), Path(f"{filename}_questions_{num_questions}.json"))
    with open(output_file_path, 'w') as output_file:
        corpus = json.dump(sampled_questions, output_file, indent=4)
