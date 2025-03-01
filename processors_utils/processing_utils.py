import re
import json
from tqdm import tqdm
import os
from pathlib import Path
from typing import Union
import csv

from component import Component

SPEC_STR = '%^&'

class DatasetProcessor(Component):
    def __init__(self, dataset_path: Path, component_name: str, split_type: str, amount: int, 
                 num_passages: Union[str, int], log: bool, working_dir: Path, output_file: Path) -> None:
        super().__init__(component_name, log, working_dir)
        self.dataset_path = Path(dataset_path)
        self.output_file = Path(output_file)
        self.split_type = split_type
        self.amount = amount
        self.num_passages = num_passages
    
    def __call__(self) -> dict:
        raw_data = self.read_dataset()
        splitted_texts_list = self.text_splitter(raw_data)
        dataset_name = self.dataset_path.stem

        # unified_corpus = [{'idx': i, 'passage': splitted_texts_dict[key]} for i, key in enumerate(splitted_texts_dict)]
        # if 'hotpotqa' in dataset_name:
        #     keys = list(splitted_texts_dict.keys())
        #     unified_corpus = [{'idx': i, 'passage': key + '\n' + ''.join(splitted_texts_dict[key])} 
        #                         for i, key in enumerate(keys)]
        # else:
        #     unified_corpus = splitted_texts_dict['data']
        #     for document in unified_corpus:
        #         document['passage'] = document.get('title', '') + '\n' + document['text']
        
        if self.num_passages == 'all':
            self.num_passages = len(splitted_texts_list)
        else:
            try:
                self.num_passages = int(self.num_passages)
            except:
                assert False, "Set 'num_passages' to an integer or 'all'"

        self.write_result(splitted_texts_list[:self.num_passages])

    def read_dataset(self):
        if os.path.isdir(self.dataset_path):
            raise ValueError("Directory")
        else:  # file
            file_extension = self.dataset_path.suffix
            
            with open(self.dataset_path, mode='r') as opened_file:
                if file_extension == '.txt':
                    # data = ''.join(opened_file.readlines()).strip('\n')  # TODO
                    data = [{"idx": idx, "passage": text_paragraph} for idx, text_paragraph 
                            in enumerate(opened_file.readlines())]

                elif file_extension == '.json':
                    data = json.load(opened_file)
                else:
                    raise ValueError(f"Unknown extension {file_extension}")
        self.write_statistics(f"amount of docs: {len(data)}")
    
        return data

    def text_splitter(self, documents):
        divided_text = []
        cur_text_fragment_list = []
        idx = 0
        if self.split_type == 'paragraph':
            for text_fragment in documents:
                if len(text_fragment['passage'].split(' ')) > 2:
                    cur_text_fragment_list.append(text_fragment['passage'].strip())
                    if len(cur_text_fragment_list) == self.amount:
                        divided_text.append({'idx': idx, 'passage': '\n'.join(cur_text_fragment_list)})
                        cur_text_fragment_list = []
                        idx += 1
                else:
                    print(f"WARNING! Fragment too short")
        elif self.split_type == 'sentence':
            text_sentences = []
            for text_fragment in documents:
                if len(text_fragment.split(' ')) > 2:
                    text_sentences.extend(text_fragment['passage'].split('.'))
            
            cur_sentences_list = []
            for sentence in text_sentences:
                if len(sentence.split(' ')) > 2:
                    cur_sentences_list.append(text_fragment['passage'].strip())
                    if len(cur_sentences_list) == self.amount:
                        divided_text.append({'idx': idx, 'passage': '\n'.join(cur_sentences_list)})
                        cur_sentences_list = []
                        idx += 1
                else:
                    print(f"WARNING! Fragment too short")
        # if type(documents) == dict:
        #     return documents
        # elif type(documents) == list:
            

        #     if self.split_type == 'sentence':
        #         dividing_by_list = ['\n', '.']
        #     elif self.split_type == 'paragraph':
        #         dividing_by_list = ['\n']

        #     for divider_char in dividing_by_list:
        #         documents = [document.strip('\n').replace(divider_char, SPEC_STR) for document in documents]

        #     for document in documents:
        #         divided_text.extend(document.split(SPEC_STR))

        #     texts_list = []

        #     index = 0
        #     for text_fragment in divided_text:
        #         if len(text_fragment) > 2:
        #             texts_list.append({"idx": index, "passage": text_fragment.strip()})
        #             index += 1
        #     # texts_list = [ for ind, text_fragment in enumerate(divided_text)]

        # else:
        #     raise ValueError()
        self.write_statistics(f"Splitted dataset length: {len(divided_text)}")
        return divided_text

    def write_result(self, processed_passages):
        writing_path = os.path.join(self.working_dir, self.output_file)
        with open(writing_path, 'w') as file:
            json.dump(processed_passages, file, indent=4)
        print('Processed passages saved to', writing_path)


class BaseProcessor(Component):
    def __init__(self, component_name: str, log: bool, working_dir: Path, 
                 input_file: Path, output_files: list) -> None:
        super().__init__(component_name, log, working_dir)
        self.input_file_path = Path(input_file)
        self.output_files = output_files

    def read_raw_triples_entities(self):
        reading_path = os.path.join(self.working_dir, self.input_file_path)
        with open(reading_path) as f:
            extracted_entities_triples = json.load(f)

        return extracted_entities_triples

    def __call__(self) -> None:
        extracted_entities_triples = self.read_raw_triples_entities()
        all_entities = []
        all_triples = []
        damaged_logs = []
        logs_dict = {}
        for extracted_sample in extracted_entities_triples:
            cur_entities = extracted_sample['extracted_entities']
            cur_triples = extracted_sample['extracted_triples']
            cur_passage = extracted_sample['text_fragment']
            full_triples, damaged_triples = self.filter_triples_by_integrity(cur_triples)
            full_entities, damaged_entities = self.filter_entities_by_integrity(cur_entities)
            if len(damaged_entities) > 0 or len(damaged_triples) > 0:
                damaged_logs.append(f"Text: {cur_passage}\ntriples: {damaged_triples}\nentities: {damaged_entities}\n\n")
            all_entities.extend(full_entities)
            all_triples.extend(full_triples)

        logs_dict['Damaged triples entities'] = damaged_logs
        all_triples = [tuple(triple) for triple in all_triples]
        
        all_entities = self.remove_non_unique(all_entities, logs_dict, 'entities')
        all_triples = self.remove_non_unique(all_triples, logs_dict, 'triples')

        # print(f"{all_entities=}\n\n{all_triples=}")
        logs_dict['Entities triples proporation'] = self.check_entities_triples_united(all_entities, all_triples)
        self.write_statistics(logs_dict)
        
        self.write_result(all_entities, all_triples)

    def remove_non_unique(self, list_of_entities_or_triples, logs_dict, things_name):
        original_length = len(list_of_entities_or_triples)
        # print(f"{list_of_entities_or_triples=}")
        filtered_list_of_entities_or_triples = list(set(list_of_entities_or_triples))
        new_length = len(filtered_list_of_entities_or_triples)
        logs_dict[f'After only unique {things_name} left'] = \
        f"Count is changed from {original_length} to {new_length}\n"
        return filtered_list_of_entities_or_triples

    def check_one_triple(self, triple):
        if len(triple) != 3: 
            return False
        for entity in triple:
            if entity == '' or type(entity) != str: 
                return False
        
        return True

    def filter_triples_by_integrity(self, triples):
        full_triples = []
        damaged_triples = []
        if triples == "":
            damaged_triples.append("")
            return full_triples, damaged_triples
        for triple in triples['triples']:
            is_full = self.check_one_triple(triple)
            if is_full:
                full_triples.append(triple)
            else:
                damaged_triples.append(triple)
        # print(f"{full_triples=}")
        # print(f"{damaged_triples=}")
        return full_triples, damaged_triples

    def filter_entities_by_integrity(self, entities):
        full_entities = []
        damaged_entities = []

        for entity in entities:
            if type(entity) != str or len(entity.split(' ')) > 6:
                damaged_entities.append(entity)
            else:
                full_entities.append(entity)

        return full_entities, damaged_entities

    def check_entities_triples_united(self, all_entities, all_triples):
        entities_from_triples = []
        for triple in all_triples:
            entities_from_triples.append(triple[0])
            entities_from_triples.append(triple[2])

        unique_triples_entities = list(set(entities_from_triples))

        not_presented_in_triples = [f'\"{item}\"' for item in all_entities if item not in unique_triples_entities]
        not_presented_in_entities = [f'\"{item}\"' for item in unique_triples_entities if item not in all_entities]

        not_presented_in_triples_str = ', '.join(not_presented_in_triples)
        not_presented_in_entities_str = ', '.join(not_presented_in_entities)
        return [f'this elements from entities are not presented in triples: {not_presented_in_triples_str}\n'
                f'this elements from triples are not presented in entities: {not_presented_in_entities_str}\n']

    def write_result(self, all_entities, all_triples):
        entities_writing_path = os.path.join(self.working_dir, Path(self.output_files[0]))
        with open(entities_writing_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for element in all_entities:
                writer.writerow([element])

        triples_writing_path = os.path.join(self.working_dir, Path(self.output_files[1]))
        with open(triples_writing_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for triple in all_triples:
                writer.writerow(triple)


        # entities_triples_list = [all_entities, all_triples]
        # for index, file_name in enumerate(self.output_files):
        #     writing_path = os.path.join(self.working_dir, Path(file_name))
        #     with open(writing_path, 'w', newline='') as file:
        #         writer = csv.writer(file)
        #         for element in entities_triples_list[index]:
        #             writer.writerow([element])
        print('Processed triples and entities are saved to ', self.working_dir)


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