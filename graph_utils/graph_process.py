import os
from pathlib import Path
import csv
from collections import defaultdict

from component import Component

class Graph:
    def __init__(self, triples_dict: dict) -> None:
        self.triples_dict = triples_dict
        # self.inv_triples_dict = inv_triples_dict

    def search(self, keys: list, depth=1):
        searched_entities = set()
        cur_entities = keys
        found_triples = []
        for cur_depth in range(depth):
            all_cur_depth_triples = []
            for entity in cur_entities:
                if entity in searched_entities:
                    continue
                cur_entity_triples = self.triples_dict.get(entity)
                if cur_entity_triples is None:
                    print(f"for key: {entity} not found any")
                else:
                    all_cur_depth_triples.extend(cur_entity_triples)
                searched_entities.add(entity)
            found_triples.append(cur_entity_triples)
            cur_entities = [triple[2] for triple in all_cur_depth_triples]

        return found_triples

class GraphConstructor(Component):
    def __init__(self, component_name, is_log, working_dir, input_file="triples.csv") -> None:
        super().__init__(component_name, is_log, working_dir)
        self.input_file = Path(input_file)

    def read_input(self):
        reading_path = os.path.join(self.working_dir, self.input_file)
        data = []
        with open(reading_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                for ind in range(len(row)):
                    row[ind] = row[ind].lower()
                data.append(row)
        return data
    
    def __call__(self) -> dict:
        triples = self.read_input()
        triples_dict = defaultdict(list)
        for triple in triples:
            first_entity = triple[0]
            second_entity = triple[2]
            triples_dict[first_entity].append(triple)
            triples_dict[second_entity].append(triple)

        return triples_dict
    
# class GraphAnswer(Component):
#     def __init__(self, component_name, is_log, working_dir, graph: Graph) -> None:
#         super().__init__(component_name, is_log, working_dir)
#         assert type(graph) == Graph
#         self.graph = graph

#     def __call__(self, query_entities) -> os.Any:
        
#         return 