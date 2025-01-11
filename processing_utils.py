import re
import json
from files_utils import read_dataset
from class_triples_entities import Triple
from files_utils import write_statistics
from tqdm import tqdm

SPEC_STR = '%^&'

def extract_json_dict(text):
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}'
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError as err:
            print(f"{err}")
            return ''
    else:
        return ''
    

def text_splitter(documents, split_type):
    divided_text = []
    if type(documents) == dict:
        return documents
    elif type(documents) == list:
        if split_type == 'sentence':
            dividing_by_list = ['\n', '.']
        elif split_type == 'paragraph':
            dividing_by_list = ['\n']

        for divider_char in dividing_by_list:
            documents = [document.strip('\n').replace(divider_char, SPEC_STR) for document in documents]

        for document in documents:
            divided_text.extend(document.split(SPEC_STR))

        texts_dict = {"classes": ["general"], "data": [{"idx": ind, "text": text_fragment} for ind, text_fragment in enumerate(divided_text)]}

    else:
        raise ValueError()
    write_statistics(f"Splitted dataset length: {len(texts_dict)}", "text_splitter")
    return texts_dict


def dataset_process(dataset_path, split_type, dataset_name, num_passages):
    corpus = read_dataset(dataset_path)
    
    prepared_corpus = text_splitter(corpus, split_type) # split_chars
    if 'hotpotqa' in dataset_name:
        keys = list(prepared_corpus.keys())
        retrieval_corpus = [{'idx': i, 'passage': key + '\n' + ''.join(prepared_corpus[key])} for i, key in enumerate(keys)]
    else:
        retrieval_corpus = prepared_corpus['data']
        for document in retrieval_corpus:
            document['passage'] = document.get('title', '') + '\n' + document['text']

    if num_passages == 'all':
        num_passages = len(retrieval_corpus)
    else:
        try:
            num_passages = int(num_passages)
        except:
            assert False, "Set 'num_passages' to an integer or 'all'"

    return retrieval_corpus[:num_passages]


def filter_triples_by_integrity(triples):
    full_triples = []
    damaged_triples = []
    print(f"{triples=}")
    for triple in triples['triples']:
        if len(triple) != 3: 
            damaged_triples.append(triple)
            continue
        for entity in triple:
            if entity == '' or type(entity) != str: 
                damaged_triples.append(triple)
                continue
        
        full_triples.append(triple)
    return full_triples, damaged_triples


def filter_entities_by_integrity(entities):
    full_entities = []
    damaged_entities = []

    for entity in entities:
        if type(entity) != str or len(entity.split(' ')) > 6:
            damaged_entities.append(entity)
        else:
            full_entities.append(entity)

    return full_entities, damaged_entities


def check_entities_triples_united(all_entities, all_triples):
    entities_from_triples = []
    for triple in all_triples:
        entities_from_triples.append(triple[0])
        entities_from_triples.append(triple[2])

    unique_triples_entities = list(set(entities_from_triples))

    not_presented_in_triples = [f'\"item\"' for item in all_entities if item not in unique_triples_entities]
    not_presented_in_entities = [f'\"item\"' for item in unique_triples_entities if item not in all_entities]

    not_presented_in_triples_str = ', '.join(not_presented_in_triples)
    not_presented_in_entities_str = ', '.join(not_presented_in_entities)
    return [f'this elements from entities are not presented in triples: {not_presented_in_triples_str}\n'
            f'this elements from triples are not presented in entities: {not_presented_in_entities_str}\n']


def base_process_entities_triples(extracted_entities_triples):
    all_entities = []
    all_triples = []
    damaged_logs = []
    logs_dict = {}
    for extracted_sample in extracted_entities_triples:
        cur_entities = extracted_sample['extracted_entities']
        cur_triples = extracted_sample['extracted_triples']
        cur_passage = extracted_sample['text_fragment']
        full_triples, damaged_triples = filter_triples_by_integrity(cur_triples)
        full_entities, damaged_entities = filter_entities_by_integrity(cur_entities)
        if len(damaged_entities) > 0 or len(damaged_triples) > 0:
            damaged_logs.append(f"Text: {cur_passage}\ntriples: {damaged_triples}\nentities: {damaged_entities}\n\n")
        all_entities.extend(full_entities)
        all_triples.extend(full_triples)

    logs_dict['Damaged triples entities'] = damaged_logs
    original_length = len(all_entities)
    all_entities = list(set(all_entities))
    new_length = len(all_entities)
    logs_dict['After only unique left'] = f"Count is changed from {original_length} to {new_length}\n"
    logs_dict['Entities triples proporation'] = check_entities_triples_united(all_entities, all_triples)
    write_statistics(logs_dict, "base_process_entities_triples")
    
    return all_entities, all_triples


def add_embeddings_to_triples_linkage(all_triples, embedding_model):
    triples_linkage_embedding = []
    for triple in tqdm(all_triples):
        cur_linkage = triple[1]
        embedding = embedding_model(cur_linkage)[0]
        triples_linkage_embedding.append(embedding)

    return triples_linkage_embedding


def merging_linkages(triples_and_embeddings, threshold=0.8):
    for index_one in range(len(triples_and_embeddings)):
        for index_second in range(index_one+1, len(triples_and_embeddings)):
            first_triple_linkage = triples_and_embeddings[index_one][1]
            second_triple_linkage = triples_and_embeddings[index_second][1]
            first_vector = triples_and_embeddings[index_one][1]