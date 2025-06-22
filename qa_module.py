import argparse
import json
import re
import os
import csv
from pathlib import Path
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm


from models_utils.llm import LLM_Phi_35, LLM_Qwen_3, LLM_T5
from graph_utils.graph_process import GraphConstructor
from models_utils.prompts_utils import TaskSplitPromptConstructor
from extractor_utils.triples_entities_extraction import Extractor


SYSTEM_PROMPT = "You're a very effective entity extraction system."
ONE_SHOT_INPUT_PROMPT = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""
ONE_SHOT_OUTPUT_PROMPT = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

USER_PROMPT_TEMPLATE = """
Question: {}

"""


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

# def process_query(prompt):
#     query_entities = extract_query_entities(prompt.strip('\n').replace('?',''))
#     print(f"{query_entities=}")
#     found_triples = STATE['graph'].search(query_entities)
#     print(f"\n{found_triples=}\n\n")
#     answer_str = 'Found triples:\n'
#     for ind, triples in enumerate(found_triples):
#         shift = ' ' * ind
#         print(triples)
#         answer_str += f'\n{shift}'.join([str(triple) for triple in triples])

#     return answer_str


# def extract_json_dict(text):
#         pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}'
#         match = re.search(pattern, text)

#         if match:
#             json_string = match.group()
#             try:
#                 json_dict = json.loads(json_string)
#                 return json_dict
#             except json.JSONDecodeError as err:
#                 print(f"{err}")
#                 return ''
#         else:
#             return ''
        

def processing_phrases(phrase):
    return re.sub('[^A-Za-z0-9 ]', ' ', phrase.lower()).strip()


def extract_query_entities(query, llm, prompt_constructor):
    named_entity_json = {"named_entities": []}
    prompt = prompt_constructor.get_task_split_prompt(task="triple", split_type='sentence')

    openie_messages = prompt.get_prompt().format_prompt(passage=query, 
                                                            named_entity_json=json.dumps(named_entity_json))

    chat_completion = llm.invoke(openie_messages.to_messages(), temperature=0, max_tokens=4096)
    response_content = chat_completion[4]['content']  # .content
    response_content = Extractor.extract_json_dict(response_content)

    response_content = str(response_content)
    triples = eval(response_content)['triples']
    # query_ner_list = [processing_phrases(p) for p in query_ner_list]
    query_entities = set()
    for triple in triples:
        entities = [triple[0], triple[2]]
        for entity in entities:
            if "text" not in entity.lower():
                query_entities.add(entity.lower())

    return query_entities


def get_answers(llm, questions_data: list, graph):
    named_entity_json = {"named_entities": []}
    prompt_constructor = TaskSplitPromptConstructor()
    prompt = prompt_constructor.get_task_split_prompt(task="triple", split_type='sentence')

    answers = {}
    supporting_facts = {}
    for question_info in tqdm(questions_data):
        question = question_info['question']
        question_id = question_info['_id']

        query_ner_messages = ChatPromptTemplate.from_messages([SystemMessage(SYSTEM_PROMPT),
                                                          HumanMessage(ONE_SHOT_INPUT_PROMPT),
                                                          AIMessage(ONE_SHOT_OUTPUT_PROMPT),
                                                          HumanMessage(USER_PROMPT_TEMPLATE.format(question))])
        # prompt = self.task_split_prompt_constructor.get_task_split_prompt(task="entity", split_type=self.split_type)
        # ner_messages = prompt.get_prompt().format_prompt(user_input=passage)
        #query_ner_messages = query_ner_prompts.format_prompt()
        chat_completion = llm.invoke(query_ner_messages, max_tokens=1536, task='ner')
        response_content = chat_completion[4]['content']   # .content
        response_content = extract_json_dict(response_content)

        if 'named_entities' not in response_content:
            query_entities = []
        else:
            query_entities = response_content['named_entities']
        # if len(query_entities) == 0:
        #     openie_messages = prompt.get_prompt().format_prompt(passage=question, 
        #                                                             named_entity_json=json.dumps(named_entity_json))

        #     query_entities = extract_query_entities(prompt.strip('\n').replace('?',''))

        # Search on graph required entities
        print("\n"+str(query_entities))
        if len(query_entities) == 0 or type(query_entities) != list or type(query_entities[0]) != str:
            answer_str = "failed"
        else:
            query_entities = [processing_phrases(p) for p in query_entities]
            found_triples = graph.search(query_entities)
            answer_str = f'Found triples:\n{found_triples[:10]}'
        #print(f"{question}\n{found_triples=}")
        answers[question_id] = answer_str
        supporting_facts[question_id] = []

    return {"answer": answers, "sp": supporting_facts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=Path, required=True)
    parser.add_argument('--llm_path', type=Path, required=True)
    parser.add_argument('--questions_path', type=Path, required=True)

    args = parser.parse_args()
    working_dir = args.working_dir
    llm_path = args.llm_path
    questions_path = args.questions_path

    with open(questions_path, mode='r') as input_file:
        questions_data = json.load(input_file)

    triples_file_path = os.path.join(Path(working_dir), Path("all_triples.csv"))
    triples = []
    with open(triples_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            for ind in range(len(row)):
                row[ind] = row[ind]  # .lower()
            triples.append(row)
    
    print(f"{Path(llm_path).name=}")
    if 'phi' in str(llm_path).lower():
        llm = LLM_Phi_35(llm_path)
    elif 't5' in str(llm_path).lower():
        llm = LLM_T5(llm_path)
    elif 'qwen' in str(llm_path).lower():
        llm = LLM_Qwen_3(llm_path)
    else:
        raise NotImplementedError()

    graph_constructor = GraphConstructor("graph_constructor", True, working_dir)
    graph = graph_constructor(triples)

    predictions = get_answers(llm, questions_data, graph)

    output_file_path = os.path.join(Path(working_dir), Path("test_qwen_fullwiki_pred.json"))
    with open(output_file_path, mode='w') as output_file:
        json.dump(predictions, output_file)


if __name__ == '__main__':
    main()
