import sys

sys.path.append('.')

#from langchain_community.chat_models import ChatOllama, ChatLlamaCpp

import argparse
import json
# from glob import glob
import re
from pathlib import Path
import os

import ipdb
from tqdm import tqdm
from langchain_openai import ChatOpenAI

from models_utils.prompts_utils import TaskSplitPromptConstructor
from models_utils.llm import init_langchain_model, LLM_Phi_35
# from models_utils.processing_utils import extract_json_dict, text_splitter, read_dataset
from component import Component

class Extractor(Component):
    def __init__(self, component_name: str, log: bool, working_dir: Path, 
                 llm_type: str, llm_path: Path, split_type: str, input_file: Path, output_file: Path) -> None:
        super().__init__(component_name, log, working_dir)
        self.llm_type = llm_type
        self.llm_path = Path(llm_path)
        self.llm = init_langchain_model(llm_type, Path(llm_path)) # LLM_Phi_35(self.llm_path)
        self.task_split_prompt_constructor = TaskSplitPromptConstructor()
        self.split_type = split_type
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)

    def __call__(self) -> None:
        processed_texts = self.read_processed_passages()
        extracted_triples_entities = self.extract_openie(processed_texts)
        self.write_result(extracted_triples_entities)

    def read_processed_passages(self):
        reading_path = os.path.join(self.working_dir, self.input_file)
        with open(reading_path, 'r') as f:
            processed_texts = json.load(f)

        return processed_texts

    def named_entity_recognition(self, passage: str):
        prompt = self.task_split_prompt_constructor.get_task_split_prompt(task="entity", split_type=self.split_type)
        ner_messages = prompt.get_prompt().format_prompt(user_input=passage)

        not_done = True

        total_tokens = 0
        response_content = '{}'

        while not_done:
            # try:
            total_tokens = 0
            chat_completion = self.llm.invoke(ner_messages.to_messages(), temperature=0, task="ner")
            response_content = chat_completion[4]['content']   # .content
            response_content = self.extract_json_dict(response_content)

            if 'named_entities' not in response_content:
                response_content = []
            else:
                response_content = response_content['named_entities']

            not_done = False
            # except Exception as e:
            #     print('Passage NER exception')
            #     print(e)

        return response_content, total_tokens

    def openie_post_ner_extract(self, passage: str, entities: list):
        named_entity_json = {"named_entities": entities}

        prompt = self.task_split_prompt_constructor.get_task_split_prompt(task="triple", split_type=self.split_type)

        openie_messages = prompt.get_prompt().format_prompt(passage=passage, 
                                                            named_entity_json=json.dumps(named_entity_json))
        total_tokens = 0
        # try:
        if isinstance(self.llm, ChatOpenAI):  # JSON mode
            chat_completion = self.llm.invoke(openie_messages.to_messages(), temperature=0, max_tokens=4096, response_format={"type": "json_object"})
            response_content = chat_completion.content
            total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
        else:
            chat_completion = self.llm.invoke(openie_messages.to_messages(), temperature=0, task="openie")
            response_content = chat_completion[4]['content']  # .content
            response_content = self.extract_json_dict(response_content)
            # response_content = str(response_content)

        # except Exception as e:
        #     print('OpenIE exception', e)
        #     return '', 0

        return response_content


    def extract_openie(self, texts_list): # client, split_type, task_split_prompt_constructor
        print(f"\n\nCreating entities")
        extracted_entities_triples = []
        chatgpt_total_tokens = 0

        for i, sample in tqdm(enumerate(texts_list)):
            passage = sample['passage']
            if len(passage) < 8:
                print(f"WARNING! Too short passage: {passage=} | SKIPPING")
                continue
            doc_entities, total_ner_tokens = self.named_entity_recognition(passage)

            try:
                doc_entities = list(set(doc_entities))
            except:
                doc_entities = doc_entities
            # print(f"Unique entities: {doc_entities}\n")
            chatgpt_total_tokens += total_ner_tokens

            triples = self.openie_post_ner_extract(passage, doc_entities)

            extracted_entities = doc_entities

            extracted_entities_triples.append({'idx': i, 'text_fragment': passage, 
                                            'extracted_entities': extracted_entities, 'extracted_triples': triples})

        return extracted_entities_triples

    def extract_json_dict(self, text):
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

    def write_result(self, extracted_entities_triples):
        writing_path = os.path.join(self.working_dir, self.output_file)
        with open(writing_path, 'w') as file:
            json.dump(extracted_entities_triples, file, indent=4)
        print('Processed raw triples and entities are saved to', writing_path)
