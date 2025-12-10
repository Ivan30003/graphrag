import re
import json
from tqdm import tqdm
import os
from pathlib import Path
from typing import Union
import csv
from collections import defaultdict
from numpy.linalg import norm
from copy import deepcopy
from nltk.corpus import stopwords

from class_triples_entities import Triple, ExpandTriple
from models_utils.embedding_model import EmbeddingModel
from component import Component

SPEC_STR = '%^&'
MAX_ENTITY_LENGTH = 8

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
        if self.is_log:
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
                if len(text_fragment['passage'].split(' ')) > 2:  # if text paragraph is not too short
                    text_sentences.extend(text_fragment['passage'].strip('.').split('.'))
            
            cur_sentences_list = []
            for sentence in text_sentences:
                if len(sentence.split(' ')) > 2: # if sentence not too short
                    cur_sentences_list.append(text_fragment['passage'].strip())
                    if len(cur_sentences_list) == self.amount:
                        divided_text.append({'idx': idx, 'passage': '\n'.join(cur_sentences_list)})
                        cur_sentences_list = []
                        idx += 1
                else:
                    print(f"WARNING! Fragment \"{sentence}\" too short")
        
        if self.is_log:
            self.write_statistics(f"Splitted dataset length: {len(divided_text)}")
        return divided_text

    def write_result(self, processed_passages):
        writing_path = os.path.join(self.working_dir, self.output_file)
        with open(writing_path, 'w') as file:
            json.dump(processed_passages, file, indent=4)
        print('Processed passages saved to', writing_path)


class BaseProcessor(Component):
    def __init__(self, component_name: str, log: bool, working_dir: Path, 
                 input_file: Path, stop_word_remove, output_files: list) -> None:
        super().__init__(component_name, log, working_dir)
        self.input_file_path = Path(input_file)
        self.output_files = output_files
        self.stop_word_remove = stop_word_remove

    def read_raw_triples_entities(self):
        reading_path = os.path.join(self.working_dir, self.input_file_path)
        with open(reading_path) as f:
            extracted_entities_triples = json.load(f)

        return extracted_entities_triples

    def __call__(self) -> None:
        extracted_entities_triples = self.read_raw_triples_entities()
        all_entities = []
        all_triples = []
        all_damaged_triples = []
        all_damaged_entities = []
        for extracted_sample in tqdm(extracted_entities_triples):
            cur_entities = extracted_sample['extracted_entities']
            cur_triples = extracted_sample['extracted_triples']
            cur_idx = extracted_sample['idx']
            full_triples, damaged_triples = self.filter_triples_by_integrity(cur_triples)
            full_entities, damaged_entities = self.filter_entities_by_integrity(cur_entities)
            all_entities.extend([entity for entity in full_entities])
            all_triples.extend([Triple(*triple, cur_idx) for triple in full_triples])
            all_damaged_entities.extend([entity for entity in damaged_entities])
            all_damaged_triples.extend([ExpandTriple(triple, cur_idx) for triple in damaged_triples])

        # logs_dict['Damaged triples count'] = f"{100*damaged_triples_count/len(all_triples)}% of damaged triples"
        # logs_dict['Damaged entities count'] = f"{100*damaged_entities_count/len(all_entities)}% of damaged entities"
        # logs_dict['All damaged triples entities'] = damaged_logs
        
        if self.stop_word_remove:
            self.remove_stop_words(all_triples)
        
        all_entities = self.remove_non_unique(all_entities, 'entities')
        all_triples = self.remove_non_unique(all_triples, 'triples')

        # print(f"{all_entities=}\n\n{all_triples=}")
        # logs_dict['Entities triples proporation'] = self.check_entities_triples_united(all_entities, all_triples)
        # self.write_statistics(logs_dict)
        
        self.write_result(all_entities, all_triples, all_damaged_entities, all_damaged_triples)

    def remove_non_unique(self, list_of_entities_or_triples, things_name):
        original_length = len(list_of_entities_or_triples)
        filtered_list_of_entities_or_triples = list(set(list_of_entities_or_triples))
        new_length = len(filtered_list_of_entities_or_triples)
        logs_dict = {f'After only unique {things_name} left': \
        f"Count is changed from {original_length} to {new_length}\n"}
        if self.is_log:
            self.write_statistics(logs_dict)
        return filtered_list_of_entities_or_triples

    def check_one_triple(self, triple):
        if len(triple) != 3:
            return False, f'Triple`s length is {len(triple)} not 3'
        for ind, entity in enumerate(triple):
            if entity == '':
                return False, f"One of Entity is an empty string (index={ind})"
            if type(entity) != str:
                return False, f"Entity type is {type(entity)} not str (index={ind})"
            if len(entity.split(" ")) > MAX_ENTITY_LENGTH:
                return False, f"Entity length in words is {len(entity.split(" "))} is too high, (index={ind})"
        
        return True, ''

    def filter_triples_by_integrity(self, triples):
        full_triples = []
        damaged_triples = []
        if triples == "":
            damaged_triples.append([""])
            return full_triples, damaged_triples
        if 'triples' in triples:
            for triple in triples['triples']:
                is_full, damage_comment = self.check_one_triple(triple)
                if is_full:
                    full_triples.append(triple)
                else:
                    damaged_triples.append(triple+[damage_comment])
        else:
            return full_triples, damaged_triples
        return full_triples, damaged_triples

    def filter_entities_by_integrity(self, entities):
        full_entities = []
        damaged_entities = []

        for entity in entities:
            if type(entity) != str or len(entity.split(' ')) > MAX_ENTITY_LENGTH:
                damaged_entities.append(entity)
            else:
                full_entities.append(entity)

        return full_entities, damaged_entities

    def remove_stop_words(self, all_triples: list[Triple], logs):
        counter_removes = 0
        for triple in all_triples:
            new_linkage_str = ' '.join([word for word in triple.linkage.split(' ') if 
                                        word not in stopwords.words('english')])
            if len(new_linkage_str) > 0:
                triple.linkage = new_linkage_str
                counter_removes += 1

        logs['stop words removing statistics'] = f"Removed stop words in {counter_removes} triples linkages from {len(all_triples)}"

    def save_entities(self, entities, path):
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            for element in entities:
                writer.writerow([element])

    def save_triples(self, triples: list[Triple], path):
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            for triple in triples:
                csv_triple = triple.table_represent
                writer.writerow(csv_triple)

    def write_result(self, all_entities, all_triples: list[Triple], damaged_entities, damaged_triples: list[ExpandTriple]):
        entities_writing_path = os.path.join(self.working_dir, Path(self.output_files[0]))
        triples_writing_path = os.path.join(self.working_dir, Path(self.output_files[1]))
        self.save_entities(all_entities, entities_writing_path)
        self.save_triples(all_triples, triples_writing_path)
        entities_writing_path = os.path.join(self.working_dir, Path(self.output_files[2]))
        triples_writing_path = os.path.join(self.working_dir, Path(self.output_files[3]))
        self.save_entities(damaged_entities, entities_writing_path)
        self.save_triples(damaged_triples, triples_writing_path)

        print('Processed triples and entities are saved to ', self.working_dir)


class LinkMerger(Component):
    def __init__(self, component_name: str, embedding_model_path: str, log: bool, working_dir: Path, 
                 input_file: Path, sim_threshold: float, output_file: Path, depth: int) -> None:
        super().__init__(component_name, log, working_dir)
        self.input_file_path = Path(input_file)
        self.output_files = output_file
        self.threshold_sim = sim_threshold
        self.embedding_model = EmbeddingModel(embedding_model_path)
        self.cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
        self.depth = depth

    def read_processed_triples(self):
        reading_path = os.path.join(self.working_dir, self.input_file_path)
        data = []
        with open(reading_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                for ind in range(len(row)):
                    row[ind] = row[ind].lower()
                data.append(row)
        return data

    def __call__(self) -> None:
        all_triples = self.read_processed_triples()
        all_links_embeddings = self.add_embeddings_to_triples_linkage(all_triples)
        merged_triples, merged_triples_stats = self.merging_linkages(all_triples, all_links_embeddings)
        linkages_before_merge = set([triple[1] for triple in all_triples])
        # print(all_triples[0][1], type(all_triples[0][1]))
        # print(merged_triples[0][1], type(merged_triples[0][1]))
        # print(merged_triples)
        linkages_after_merge = set([triple[1] for triple in merged_triples])
        merged_triples_stats['length changes'] = f"before merging: {len(linkages_before_merge)} | after merging: {len(linkages_after_merge)}"
        self.write_statistics(merged_triples_stats)
        self.write_result(merged_triples)


    def add_embeddings_to_triples_linkage(self, all_triples):
        triples_linkages = [triple[1].strip().strip("'") for triple in all_triples]
        all_triples_embedding = self.embedding_model(triples_linkages)

        return all_triples_embedding

    def merging_linkages(self, all_triples, all_links_embeddings):
        EXCLUDING_LIST = ["was", "was a", "was an", "is", "is a", "is an", "were", "are", ]
        merged_triples_stats = {"merged_triples": dict()}
        sim_link_dict = defaultdict(list)
        for index_one in tqdm(range(len(all_links_embeddings))):
            first_triple_linkage = all_triples[index_one][1]
            if first_triple_linkage in EXCLUDING_LIST:
                continue
            for index_second in range(index_one+1, len(all_links_embeddings)):
                second_triple_linkage = all_triples[index_second][1]
                first_vector = all_links_embeddings[index_one]
                second_vector = all_links_embeddings[index_second]
                if self.cos_sim(first_vector, second_vector) > self.threshold_sim:
                    # print(f"SIMILAR: {all_triples[index_one][1]} | {all_triples[index_second][1]}")
                    sim_link_dict[index_one].append(index_second)

        # for key in sim_link_dict:
        #     print(f"KEY: {key}")
        #     for element in sim_link_dict[key]:
        #         print(f"    {element}")
        # cur_links = list(sim_link_dict.keys())
        # center_united_links = defaultdict(list)
        searched_links = set()
        merged_triples = deepcopy(all_triples)
        for key in sim_link_dict:
            # if key in searched_links:
            #     continue
            main_link = merged_triples[key][1]
            merged_triples_stats['merged_triples'][main_link] = []
            for index in sim_link_dict[key]:
                if key in searched_links:
                    continue
                merged_triples_stats['merged_triples'][main_link].append(merged_triples[index][1])
                merged_triples[index][1] = main_link  # replace linkage with similar one
                searched_links.add(index)

        return merged_triples, merged_triples_stats


        # for cur_depth in range(self.depth):
        #     all_cur_depth_triples = []
        #     for entity in cur_entities:
        #         if entity in searched_entities:
        #             continue
        #         searched_entities.add(entity)
        #         cur_entity_triples = sim_link_dict.get(entity)
        #         if cur_entity_triples is None:
        #             print(f"for key: {entity} not found any")
        #             continue
        #         else:
        #             all_cur_depth_triples.extend(cur_entity_triples)
        #     found_triples.append(all_cur_depth_triples)


    def write_result(self, triples: list):
        triples_writing_path = os.path.join(self.working_dir, Path(self.output_files))
        with open(triples_writing_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for triple in triples:
                writer.writerow(triple)

        print('Merged triples are saved to ', self.working_dir)