from tqdm import tqdm
from pathlib import Path
from component import Component
import csv


HUMAN_ANNOTATED_LINKS_REVERSE_DICT = {
    "cannot be replicated by": "cannot replicate",
    "has classics like": "is presented on",
    "has a cast including": "starred in the film",
    "has a population addicted to": "have great influence on population of",
    "welcomed their second child": "is a second child, which is welcomed by",
    "wins over" : "loses to",
    "requires": "are requered for",
    "offer": "is offered by",
    "led to": "is caused by",
    "include": "are included in"
}


class TripleReverser(Component):
    def __init__(self, component_name: str, log: bool, working_dir: Path, 
                 input_file: Path, output_file: Path, llm_type: str, 
                 llm_path: Path) -> None:
        super().__init__(component_name, log, working_dir)
        if llm_path or llm_type == 'openai':
            pass
            ##TODO self.triple_reverser = ""
        else:
            self.triple_reverser = self.add_reversed_triples_by_human_annotated_dict
        self.input_file_path = Path(input_file)
        self.output_files = output_file

    def read_processed_triples(self):
        reading_path = self.working_dir / self.input_file_path
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
        added_triples = self.add_reversed_triples(all_triples)
        # merged_triples, merged_triples_stats = self.merging_linkages(all_triples, all_links_embeddings)
        # linkages_before_merge = set([triple[1] for triple in all_triples])
        # linkages_after_merge = set([triple[1] for triple in merged_triples])
        merged_triples_stats = {'new_added_triples_num': len(added_triples)}
        self.write_statistics(merged_triples_stats)
        self.write_result(all_triples+added_triples)

    def add_reversed_triples(self, all_triples):
        added_triples = self.triple_reverser(all_triples)

        return added_triples

    def add_reversed_triples_by_human_annotated_dict(self, all_triples):
        added_triples = []
        for triple in all_triples:
            if triple[1] in HUMAN_ANNOTATED_LINKS_REVERSE_DICT:
                reversed_triple = [triple[2], HUMAN_ANNOTATED_LINKS_REVERSE_DICT[triple[1]], triple[0]]
                added_triples.append(reversed_triple)

        return added_triples



    def write_result(self, triples: list):
        triples_writing_path = self.working_dir / Path(self.output_files)
        with open(triples_writing_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for triple in triples:
                writer.writerow(triple)

        print('Merged triples are saved to ', self.working_dir)