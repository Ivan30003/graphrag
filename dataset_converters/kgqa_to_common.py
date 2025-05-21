import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_json_path', type=str)
    parser.add_argument('--output_json_path', type=str)
    parser.add_argument('--num_passages', type=str)
    args = parser.parse_args()
    input_path = args.dataset_json_path
    output_path = args.output_json_path
    passages_num = args.num_passages

    with open(input_path, 'r') as input_file:
        corpus = json.load(input_file)
    # keys = list(corpus.keys())
    if passages_num == 'all':
        passages_num = len(corpus)
    else:
        passages_num = int(passages_num)
    print(f"{passages_num} num of samples will be taken")
    retrieval_corpus = []
    for sample in corpus:
        unique_id = sample['id']
        passages_objs = sample['wiki_data']
        if len(passages_objs) > 1:
            print("\n\nLONGER")
            for ind in range(len(passages_objs)):
                print(f"{ind}: {passages_objs[ind]=}\n")
            raise ValueError()
        for passage_obj in passages_objs:
            if len(passage_obj) != 1 and type(passage_obj) != dict:
                raise f"DATA is broken on {passages_objs=}"
            text = list(passage_obj.values())[0]
            passage_dict = {'idx': unique_id, 'passage': text}
            retrieval_corpus.append(passage_dict)

    print(f"Total num of text paragraphs is {len(retrieval_corpus)} (can be different from num of samples)")
    with open(output_path, 'w') as output_file:
        corpus = json.dump(retrieval_corpus, output_file)