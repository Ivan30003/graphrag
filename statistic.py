import argparse
from tqdm import tqdm
from pathlib import Path
from nltk.corpus import stopwords
import json
import csv
import matplotlib.pyplot as plt

from files_utils import read_yaml_file, read_csv_file, read_json
from class_triples_entities import Triple, ExpandTriple


class ParagraphStats:
    def __init__(self, working_dir, output_file_name) -> None:
        pass

    def __call__(self) -> None:
        pass

class ExtractorStats:
    def __init__(self, working_dir, input_file_path) -> None:
        self.input_file_path = Path(input_file_path)
        self.working_dir = Path(working_dir)

    def read_input(self) -> list:
        reading_path = self.working_dir / self.input_file_path
        with open(reading_path) as f:
            extracted_entities_triples = json.load(f)

        return extracted_entities_triples

    def count_words_str(self, text: str):
        non_stop_words = [word for word in text.split(' ') if word not in stopwords.words('english')]
        return len(non_stop_words)

    def __call__(self) -> None:
        extracted_entities_triples = self.read_input()
        total_entities = 0
        total_entities_words = 0
        total_triples = 0
        total_triples_words = 0
        total_paragraphs = len(extracted_entities_triples)
        total_paragraphs_words = 0
        paragraph_entities_words_proportion_list = []
        paragraph_triples_words_proportion_list = []
        for sample in tqdm(extracted_entities_triples):
            # one paragraph stats
            entities = sample['extracted_entities']
            entities_str = ' '.join(set(entities))
            entities_words = self.count_words_str(entities_str)
            if type(sample['extracted_triples']) == dict and 'triples' in sample['extracted_triples']:
                triples = sample['extracted_triples']['triples']
            else:
                triples = [[]]
            try:
                triples_str = ' '.join([' '.join(triple).strip() for triple in triples])
                triples_words = self.count_words_str(triples_str)
                paragraph_words = self.count_words_str(sample['text_fragment'])
                
                # save total
                total_entities += len(entities)
                total_triples += len(triples)
                total_entities_words += entities_words
                total_triples_words += triples_words
                total_paragraphs_words += paragraph_words
                paragraph_entities_words_proportion_list.append(entities_words/paragraph_words)
                paragraph_triples_words_proportion_list.append(triples_words/paragraph_words)
            except Exception as err:
                print(f"ERROR! {err}")


        total_entities_proportion = total_entities / total_paragraphs
        total_triples_proportion = total_triples / total_paragraphs

        total_words_entities_proportion = total_entities_words / total_paragraphs_words
        total_words_triples_proportion = total_triples_words / total_paragraphs_words
        assert len(paragraph_entities_words_proportion_list) == len(paragraph_triples_words_proportion_list)

        self.write_statistics(total_entities_proportion, total_triples_proportion, 
                              total_words_entities_proportion, total_words_triples_proportion,
                              paragraph_entities_words_proportion_list,
                              paragraph_triples_words_proportion_list)
        
        self.draw_and_save_hist(paragraph_entities_words_proportion_list, "entities")
        self.draw_and_save_hist(paragraph_triples_words_proportion_list, "triples")
        
    def write_statistics(self, total_entities_proportion, total_triples_proportion, 
                              total_words_entities_proportion, total_words_triples_proportion,
                              paragraph_entities_words_proportion_list,
                              paragraph_triples_words_proportion_list):
        writing_path = self.working_dir / Path("extractor_statistics.csv")
        with open(writing_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # num statistics
            headers_row = ["total entities proportion", "total triples proportion", 
                           "total words entities proportion", "total words triples proportion"]
            values_row = [total_entities_proportion, total_triples_proportion, 
                          total_words_entities_proportion, total_words_triples_proportion]
            writer.writerow(headers_row)
            writer.writerow(values_row)
            # hist statistics
            headers_row = ["paragraph_entities_words_proportion", "paragraph_triples_words_proportion"]
            writer.writerow(headers_row)
            for entity, triple in zip(paragraph_entities_words_proportion_list, 
                                      paragraph_triples_words_proportion_list):
                writer.writerow([entity, triple])

    def draw_and_save_hist(self, data: list, x_label_add):
        x_label = f"{x_label_add} paragraphs words proportion"
        plt.hist(data, bins=20)
        plt.title(f"{x_label} distribution")
        plt.xlabel(x_label)
        plt.ylabel("num paragraphs")
        plt.grid(True)
        if "entities" in x_label_add:
            # plt.xlim(0.0, 1.0)
            # plt.ylim(0, 180)
            pass
        else:
            # plt.xlim(0.0, 6.0)
            # plt.ylim(0, 400)
            pass
        writing_path = self.working_dir / Path(f"{x_label_add}_distribution.png")
        plt.savefig(writing_path)
        plt.close()


class BaseProcessorStats:
    def __init__(self, working_dir, output_file_names) -> None:
        self.working_dir = Path(working_dir)
        self.output_file_names = output_file_names  # [all_entities, all_triples, damaged_entities, damaged_triples]

    def calculate_triples_entities_quality(self, all_triples):
        total_entities_words_count = 0
        for triple in all_triples:
            entity_first = triple[0]
            entity_second = triple[2]
            entity_first_word_count = len([word for word in entity_first.split(' ') 
                                           if word not in stopwords.words('english')])
            entity_second_word_count = len([word for word in entity_second.split(' ') 
                                            if word not in stopwords.words('english')])
            total_entities_words_count += (entity_first_word_count + entity_second_word_count)
        
        return total_entities_words_count / (2*len(all_triples))


    def check_entities_triples_united(self, all_entities: list, all_triples: list[Triple]):
        entities_from_triples = []
        for triple in all_triples:
            entities_from_triples.append(triple.first_entity)
            entities_from_triples.append(triple.second_entity)

        unique_triples_entities = set(entities_from_triples)
        intersection = list(unique_triples_entities & set(all_entities))
        not_presented_in_triples = [item for item in all_entities if item not in unique_triples_entities]
        not_presented_in_entities = [item for item in unique_triples_entities if item not in all_entities]

        entities_presence_intersection_prop = len(intersection) / len(all_entities)
        triples_presence_intersection_prop = len(intersection) / len(unique_triples_entities)
        entities_own_triples_prop = len(not_presented_in_entities) / len(unique_triples_entities)
        entities_own_entities_prop = len(not_presented_in_triples) / len(all_entities)
        return entities_presence_intersection_prop, triples_presence_intersection_prop, \
        entities_own_entities_prop, entities_own_triples_prop
    
    def read_input(self):
        reading_path = self.working_dir / Path(self.output_file_names[0])
        all_entities = read_csv_file(reading_path)
        reading_path = self.working_dir / Path(self.output_file_names[1])
        all_triples = read_csv_file(reading_path)
        reading_path = self.working_dir / Path(self.output_file_names[2])
        damaged_entities = read_csv_file(reading_path)
        reading_path = self.working_dir / Path(self.output_file_names[3])
        damaged_triples = read_csv_file(reading_path)

        return all_entities, all_triples, damaged_entities, damaged_triples

    def __call__(self) -> None:
        all_entities, all_triples, damaged_entities, damaged_triples = self.read_input()
        converted_all_triples = [Triple(*triple[:3], triple[-1]) for triple in all_triples]
        damaged_triples_proportion = len(damaged_triples) / (len(converted_all_triples) + len(damaged_triples))

        if len(all_entities) > 0:
            converted_all_entities = [entity[0] for entity in all_entities]
            damaged_entities_proportion = len(damaged_entities) / (len(all_entities) + len(damaged_entities))
            entities_presence_intersection_prop, triples_presence_intersection_prop, \
            entities_own_entities_prop, entities_own_triples_prop = self.check_entities_triples_united(converted_all_entities, 
                                                                                                    converted_all_triples)
        else:
            damaged_entities_proportion = 0.0
            entities_presence_intersection_prop = 0.0
            triples_presence_intersection_prop = 0.0
            entities_own_entities_prop = 0.0
            entities_own_triples_prop = 0.0

        mean_triples_entities_words_count =  self.calculate_triples_entities_quality(all_triples)

        self.write_statistics(damaged_triples_proportion, damaged_entities_proportion, 
                              entities_presence_intersection_prop, triples_presence_intersection_prop,
                            entities_own_entities_prop, entities_own_triples_prop, mean_triples_entities_words_count)

    def write_statistics(self, damaged_triples_proportion, damaged_entities_proportion,
                         entities_presence_intersection_prop, triples_presence_intersection_prop,
                            entities_own_entities_prop, entities_own_triples_prop, mean_triples_entities_words_count):
        writing_path = self.working_dir / Path("base_processor_statistics.csv")
        with open(writing_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # num statistics
            headers_row = ["damaged triples proportion", "damaged entities proportion", 
                           "entities presented in intersection_proportion", 
                           "triples entities presented in intersection proportion",
                           "entities entities which are not in triples proportion",
                           "triples entities which are not in entities proportion", 
                           "mean words in entity in all triples"]
            values_row = [damaged_triples_proportion, damaged_entities_proportion, 
                          entities_presence_intersection_prop, triples_presence_intersection_prop,
                          entities_own_entities_prop, entities_own_triples_prop, mean_triples_entities_words_count]
            for header, value in zip(headers_row, values_row):
                writer.writerow([header, value])


class HierarchyStats:
    def __init__(self) -> None:
        pass


PARTS_STATS_DICT = {"dataset_processor": ParagraphStats, 
                    "extractor": ExtractorStats, 
                    "base_processor": BaseProcessorStats,
                    "hierarchy_module": HierarchyStats}


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_path', type=Path)
    parser.add_argument('--config_path', type=Path, required=True)

    args = parser.parse_args()

    # dataset_path = args.dataset_path
    config_path = args.config_path
    config = read_yaml_file(config_path)
    statistics_pipeline = []
    working_dir = config['working_dir']
    for part in config['pipeline']['parts']:
        part_class = PARTS_STATS_DICT.get(part)
        if part_class:
            params = config['pipeline']['parts'][part]
            output_file_name = params.get('output_file')
            if output_file_name:
                stat_module = part_class(working_dir, output_file_name)
            else:
                output_file_names = params['output_files']
                stat_module = part_class(working_dir, output_file_names)
            statistics_pipeline.append(stat_module)
        else:
            print(f"'{part_class}' NOT IMPLEMENTED WARNING!")

    for part in statistics_pipeline:
        part()



if __name__ == "__main__":
    main()