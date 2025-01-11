import sys

sys.path.append('.')

#from langchain_community.chat_models import ChatOllama, ChatLlamaCpp

import argparse
import json
from glob import glob

import ipdb
from tqdm import tqdm
from langchain_openai import ChatOpenAI

from prompts_utils import TaskSplitPromptConstructor
from processing_utils import extract_json_dict, text_splitter, read_dataset
from class_triples_entities import Triple


def named_entity_recognition(passage: str, client, split_type, task_split_prompt_constructor):
    prompt = task_split_prompt_constructor.get_task_split_prompt(task="entity", split_type=split_type)
    ner_messages = prompt.get_prompt().format_prompt(user_input=passage)

    not_done = True

    total_tokens = 0
    response_content = '{}'

    while not_done:
        # try:
        total_tokens = 0
        chat_completion = client.invoke(ner_messages.to_messages(), temperature=0)
        response_content = chat_completion[4]['content']   # .content
        response_content = extract_json_dict(response_content)

        if 'named_entities' not in response_content:
            response_content = []
        else:
            response_content = response_content['named_entities']

        not_done = False
        # except Exception as e:
        #     print('Passage NER exception')
        #     print(e)

    return response_content, total_tokens


def openie_post_ner_extract(passage: str, entities: list, client, split_type, task_split_prompt_constructor):
    named_entity_json = {"named_entities": entities}

    prompt = task_split_prompt_constructor.get_task_split_prompt(task="triple", split_type=split_type)

    openie_messages = prompt.get_prompt().format_prompt(passage=passage, named_entity_json=json.dumps(named_entity_json))
    total_tokens = 0
    # try:
    if isinstance(client, ChatOpenAI):  # JSON mode
        chat_completion = client.invoke(openie_messages.to_messages(), temperature=0, max_tokens=4096, response_format={"type": "json_object"})
        response_content = chat_completion.content
        total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
    else:
        chat_completion = client.invoke(openie_messages.to_messages(), temperature=0, max_tokens=4096)
        response_content = chat_completion[4]['content']  # .content
        response_content = extract_json_dict(response_content)
        # response_content = str(response_content)

    # except Exception as e:
    #     print('OpenIE exception', e)
    #     return '', 0

    return response_content


def extract_openie(json_set, client, split_type, task_split_prompt_constructor):
    print(f"\n\nCreating entities")
    extracted_entities_triples = []
    chatgpt_total_tokens = 0

    for i, sample in tqdm(enumerate(json_set)):
        # print(f"\n{'*'*50}\n")
        passage = sample['passage']
        if len(passage) < 8:
            print(f"WARNING! Too short passage: {passage=} | SKIPPING")
            continue
        # print(f"Current text: \n{passage}<END>\n")
        doc_entities, total_ner_tokens = named_entity_recognition(passage, client, split_type, task_split_prompt_constructor)
        # print(f"Entities: {doc_entities}\n")
        try:
            doc_entities = list(set(doc_entities))
        except:
            doc_entities = doc_entities
        # print(f"Unique entities: {doc_entities}\n")
        chatgpt_total_tokens += total_ner_tokens

        triples = openie_post_ner_extract(passage, doc_entities, client, split_type, task_split_prompt_constructor)

        extracted_entities = doc_entities

        # try:
        #     extracted_triples = eval(triples)["triples"]
        # except Exception as er:
        #     print(f'ERROR: {er}')
        #     print(triples)
        #     extracted_triples = []

        # print(f"Triples: {r['extracted_triples']}")  # [0]['triples']
        extracted_entities_triples.append({'idx': i, 'text_fragment': passage, 
                                           'extracted_entities': extracted_entities, 'extracted_triples': triples})

    return extracted_entities_triples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    # parser.add_argument('--run_ner', action='store_true')
    parser.add_argument('--num_passages', type=str, default='10')
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    parser.add_argument('--split_type', choices=['sentence', 'paragraph'], type=str, default='')
    parser.add_argument('--output_path', type=str, help='Specific path result write to')

    args = parser.parse_args()

    dataset_path = args.dataset_path
    num_passages = args.num_passages
    model_name_or_path = args.model_name
    # print(f"{args.split_chars.replace()=}")
    corpus, dataset_type = read_dataset(args.dataset_path)
    print()
    splitted_corpus = text_splitter(corpus, args.split_type) # split_chars
    if 'hotpotqa' in dataset_path:
        keys = list(splitted_corpus.keys())
        retrieval_corpus = [{'idx': i, 'passage': key + '\n' + ''.join(splitted_corpus[key])} for i, key in enumerate(keys)]
    else:
        retrieval_corpus = splitted_corpus['data']
        for document in retrieval_corpus:
            document['passage'] = document.get('title', '') + '\n' + document['text']

    if num_passages == 'all':
        num_passages = len(retrieval_corpus)
    else:
        try:
            num_passages = int(num_passages)
        except:
            assert False, "Set 'num_passages' to an integer or 'all'"

    splitting_type = args.split_type
    arg_str = dataset_type + model_name_or_path.replace('/', '_') + f'_{num_passages}' + f"_{splitting_type}"

    client = init_langchain_model(args.llm, model_name_or_path)  # LangChain model
    task_split_prompt_constructor = TaskSplitPromptConstructor()
    
    extracted_subset = retrieval_corpus[:num_passages]
    new_json, all_entities, all_triples = extract_openie(extracted_subset, client, splitting_type, 
                                                         task_split_prompt_constructor)

    lm_total_tokens = 0

    # for output in outputs:
    #     new_json.extend(output)
    #     # all_entities.extend(output[1])
    #     # lm_total_tokens += output[2]

    already_done = False
    if not (already_done):
        # avg_ent_chars = np.mean([len(e) for e in all_entities])
        # avg_ent_words = np.mean([len(e.split()) for e in all_entities])

        # Current Cost
        approx_total_tokens = (len(retrieval_corpus) / num_passages) * lm_total_tokens

        extra_info_json = {"docs": new_json,
                           "all_entities": all_entities,
                           "all_triples": all_triples,
                           # "avg_ent_chars": avg_ent_chars,
                           # "avg_ent_words": avg_ent_words,
                           "num_tokens": lm_total_tokens,
                           "approx_total_tokens": approx_total_tokens,
                           }
        output_path = f'{args.output_path}openie_results_{arg_str}.json'
        json.dump(extra_info_json, open(output_path, 'w'), indent=4)
        print('OpenIE saved to', output_path)
