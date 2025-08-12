from typing import Any
from component import Component
from pathlib import Path
import os
import csv
from collections import defaultdict


ATTRIBUTE_KEY_WORD = "has"


class HierarchyModule(Component):
    def __init__(self, component_name, log, working_dir, 
                 input_file: Path, output_file: str) -> None:
        super().__init__(component_name, log, working_dir)
        self.input_file = input_file
        self.output_files = output_file

    def __call__(self) -> None:
        triples = self.read_processed_triples()
        attributes_dict, attributes_pair_counts = self.locate_attributes(triples)
        parents_dict, parent_child_counts = self.locate_parents(triples)
        logs_dict = {"counts": f"attributes found: {attributes_pair_counts} | parents childs relations found: {parent_child_counts}"}
        # print(f"{attributes_dict=}")
        # print(f"\n\n{parents_dict=}")

        attributes_nouns = set(attributes_dict.keys())
        parents_nouns = set(parents_dict.keys())
        # print(f"\n{len(attributes_nouns)=} | {len(parents_nouns)=}")
        intersect = list(attributes_nouns & parents_nouns)
        logs_dict["pairs parent -> child attributes counts"] = len(intersect)

        # Enriching triples with new ones
        new_triples = []
        for key in intersect: 
            childs = parents_dict[key]
            parent_attributes_words = attributes_dict[key]
            for child in childs:
                child_attributes_words = attributes_dict[child]
                for parent_attributes_word in parent_attributes_words:
                    if parent_attributes_word not in child_attributes_words:
                        new_triples.append([child, ATTRIBUTE_KEY_WORD, parent_attributes_word, "logic-based"])

        logs_dict["new triples appeared due to attribute passing"] = len(new_triples)
        triples.extend(new_triples)
        self.write_result(triples)
        self.write_statistics(logs_dict)

    def read_processed_triples(self):
        reading_path = os.path.join(self.working_dir, self.input_file)
        data = []
        with open(reading_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                for ind in range(len(row)):
                    row[ind] = row[ind].lower()
                data.append(row)
        return data
    
    def locate_attributes(self, triples):
        count = 0
        attributes_dict = defaultdict(list)
        for triple in triples:
            if triple[1] in ['has', 'have']:
                count += 1
                attributes_dict[triple[0]].append(triple[2])

        return attributes_dict, count
    
    def locate_parents(self, triples):
        count = 0
        parents_dict = defaultdict(list)
        for triple in triples:
            if triple[1] in ['is', 'is a', 'are', 'work as', "works as"]:
                count += 1
                parents_dict[triple[2]].append(triple[0])
            if triple[1] in ['includes', 'include']:
                count += 1
                parents_dict[triple[0]].append(triple[2])

        return parents_dict, count
    
    def write_result(self, triples: list):
        triples_writing_path = os.path.join(self.working_dir, Path(self.output_files))
        with open(triples_writing_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for triple in triples:
                writer.writerow(triple)

        print('Enriched triples are saved to ', self.working_dir)