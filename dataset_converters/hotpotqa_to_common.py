import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_json_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    input_path = args.dataset_json_path
    output_path = args.output_path

    with open(input_path, 'r') as input_file:
        corpus = json.load(input_file)
    keys = list(corpus.keys())
    retrieval_corpus = [{'idx': i, 'passage': key + '\n' + ''.join(corpus[key])} for i, key in enumerate(keys)]

    with open(output_path, 'w') as output_file:
        corpus = json.dump(retrieval_corpus, output_file)
