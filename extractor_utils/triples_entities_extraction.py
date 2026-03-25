import sys

sys.path.append('.')

import json
import re
from pathlib import Path
import os

from tqdm import tqdm
from langchain_openai import ChatOpenAI
import timeout_decorator

from models_utils.prompts_utils import TaskSplitPromptConstructor
from models_utils.llm import init_langchain_model
# from models_utils.processing_utils import extract_json_dict, text_splitter, read_dataset
from component import Component


SAVE_KEY_WORD = "_save_step_"  # extracted_entities_triples_save_step_500.json
MIN_JSON_LENGTH = 3

class Extractor(Component):
    def __init__(self, component_name: str, log: bool, working_dir: Path, llm_type: str, 
                 llm_path: Path, split_type: str, input_file: Path, output_file: Path, save_each_steps=0,
                 simplier_pattern=False, separate_entities_extraction_step=False, seed=52, max_attempts=1) -> None:
        super().__init__(component_name, log, working_dir)
        self.llm_type = llm_type
        self.llm_path = Path(llm_path)
        self.llm = init_langchain_model(llm_type, Path(llm_path), seed=seed)
        self.task_split_prompt_constructor = TaskSplitPromptConstructor()
        self.split_type = split_type
        self.simplier_pattern = simplier_pattern
        self.save_each_steps = save_each_steps
        self.separate_entities_extraction_step = separate_entities_extraction_step
        self.max_attempts = max_attempts
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)

    def __call__(self) -> None:
        processed_texts = self.read_processed_passages()

        # Check if save was done
        main_output_name = self.output_file.stem
        saves_pattern = f"{main_output_name}{SAVE_KEY_WORD}*.json"
        save_files = [file_name for file_name in Path(self.working_dir).glob(saves_pattern)]
        start_index = 0
        saved_extracted_triples_entities = []
        if len(save_files) > 0:
            saved_extracted_triples_entities, start_index = self.load_save(save_files)

        extracted_triples_entities = self.extract_openie(saved_extracted_triples_entities, processed_texts, start_index)
        # saved_extracted_triples_entities.extend(extracted_triples_entities)
        self.write_result(extracted_triples_entities)

    def read_processed_passages(self):
        reading_path = os.path.join(self.working_dir, self.input_file)
        with open(reading_path, 'r') as f:
            processed_texts = json.load(f)

        return processed_texts

    def load_save(self, save_files: list):
        save_files.sort(key=lambda x: int(str(x).split('.')[-2].split('_')[-1]))
        last_save_path = save_files[-1]
        reading_path = os.path.join(self.working_dir, last_save_path)
        with open(reading_path, 'r') as f:
            processed_entities_triples = json.load(f)
        print(f"LOAD save from file: {reading_path}")
        assert int(str(last_save_path).split('.')[-2].split('_')[-1]) == len(processed_entities_triples)
        return processed_entities_triples, len(processed_entities_triples)

    def is_repeat(self, entities):
        if type(entities) == dict:
            if len(entities['triples']) < 3:
                return False
            for i in range(len(entities['triples'])-2):
                first_triple_str = ' | '.join(entities['triples'][i])
                second_triple_str = ' | '.join(entities['triples'][i+1])
                third_triple_str = ' | '.join(entities['triples'][i+2])
                if first_triple_str == second_triple_str:
                    return True
                if first_triple_str == third_triple_str:
                    return True
        elif type(entities) == list:
            if len(entities) < 3:
                return False
            for i in range(len(entities)-2):
                if entities[i] == entities[i+1] and entities[i+1] == entities[i+2]:
                    return True
        else:
            raise ValueError(f"{type(entities)=}")
        return False

    def named_entity_recognition(self, passage: str):
        prompt = self.task_split_prompt_constructor.get_task_split_prompt(task="entity", split_type=self.split_type)
        ner_messages = prompt.get_prompt().format_prompt(user_input=passage)

        total_tokens = 0
        response_content = '{}'
        temperature = 0.0

        for attempt in range(self.max_attempts):
            chat_completion = self.llm.invoke(ner_messages.to_messages(), temperature=temperature, task="ner")
            raw_response_content = chat_completion[4]['content']   # .content
            try:
                response_content = self.extract_json_dict(raw_response_content)
                response_content = response_content['named_entities']
                is_repeat = self.is_repeat(response_content)
            except Exception as err:
                print(f"ERROR: parsing problem: {err}")
                response_content = []
            if len(response_content) > 0 and not is_repeat:
                break
            temperature = 0.6

        return response_content

    def openie_post_ner_extract(self, passage: str, entities: list):
        if len(entities) > 0:
            named_entity_json = {"named_entities": entities}
            named_entity_json_str = json.dumps(named_entity_json)
        else:
            named_entity_json_str = ""

        prompt_constructor = self.task_split_prompt_constructor.get_task_split_prompt(task="triple", split_type=self.split_type)

        openie_messages = prompt_constructor.get_prompt().format_prompt(passage=passage, 
                                                            named_entity_json=named_entity_json_str)
        # try:

        temperature = 0.0

        for attempt in range(self.max_attempts):
            chat_completion = self.llm.invoke(openie_messages.to_messages(), temperature=temperature, task="ner")
            raw_response_content = chat_completion[4]['content']   # .content
            try:
                response_content = self.extract_json_dict(raw_response_content)
                response_content = response_content['triples']
                is_repeat = self.is_repeat(response_content)
            except Exception as err:
                print(f"ERROR: {err}")
                response_content = []
            if len(response_content) > 0 and not is_repeat:
                break
            
            temperature = 0.6
            print(f"REPEAT")

        return response_content


    def extract_openie(self, saved_extracted_triples_entities: list, texts_list, start_index=0): # client, split_type, task_split_prompt_constructor
        extracted_entities_triples = []
        chatgpt_total_tokens = 0
        index = 0
        for sample in tqdm(texts_list[start_index:]):
            passage = sample['passage']
            if len(passage) < 8:
                print(f"WARNING! Too short passage: {passage=} | SKIPPING")
                continue
            if self.separate_entities_extraction_step:
                doc_entities = self.named_entity_recognition(passage)

                try:
                    doc_entities = sorted(list(set(doc_entities)))
                except:
                    doc_entities = [doc_entities]
            else:
                doc_entities = []
            # print(f"Unique entities: {doc_entities}\n")
            cur_index = start_index + index
            if cur_index in [1147]:
                print(f"{cur_index} - BAD SAMPLE:\n{passage}")
                triples = []
            else:
                triples = self.openie_post_ner_extract(passage, doc_entities)

            extracted_entities = doc_entities

            extracted_entities_triples.append({'idx': sample.get('idx'), 'dataset_idx': sample.get('id'), 
                                               'text_fragment': passage, 
                                                'extracted_entities': extracted_entities,
                                                'extracted_triples': triples})
            
            # saving
            if self.save_each_steps != 0:
                if len(extracted_entities_triples) % self.save_each_steps == 0:
                    # saved_extracted_triples_entities.extend(extracted_entities_triples)
                    self.write_result(saved_extracted_triples_entities + extracted_entities_triples, ready=False)

            index+=1
        return saved_extracted_triples_entities + extracted_entities_triples

    @timeout_decorator.timeout(10)
    def extract_json_dict(self, text):
        if self.simplier_pattern:
            pattern = re.compile(r'(\{.*?\})')
        else:
            pattern = re.compile(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}')
        match = re.search(pattern, text)

        if match:
            json_string = match.group()
            try:
                json_dict = json.loads(json_string)
                return json_dict
            except json.JSONDecodeError as err:
                return ''
        else:
            return ''

    def write_result(self, extracted_entities_triples, ready=True):
        if ready:
            writing_path = os.path.join(self.working_dir, self.output_file)
            print('Processed raw triples and entities are saved to', writing_path)
        else:
            writing_path = os.path.join(self.working_dir, f"{self.output_file.stem}{SAVE_KEY_WORD}{len(extracted_entities_triples)}.json")
            print('Processed PART of raw triples and entities are saved to', writing_path)
        with open(writing_path, 'w') as file:
            json.dump(extracted_entities_triples, file, indent=4)
        
