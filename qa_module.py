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
from graph_utils.graph_process import GraphConstructor, Graph
from models_utils.prompts_utils import TaskSplitPromptConstructor
from extractor_utils.triples_entities_extraction import Extractor


SYMBOL_COUNT_LIMIT = 16000
ATTEMPTS = 3


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


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


SYSTEM_PROMPT_QA = "You're thorough assistant responding to questions, based on retrieved context"
USER_PROMPT_TEMPLATE_QA = """
Goal:
1) Provide clear and accurate response: carefully review and verify the retrieved data, and integrate any relevant necessary knowledge to comprehensively address user's question.
2) Do not fabricate information: if you are unsure of answer just say so.
3) Do not include details, not supported by the provided evidence
4) Place your short answer to the question after words "{answer_key_word}"

Context:
{context}

User's question:
{question}
"""
KEY_WORD = "So the answer is:"


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


def get_context_str_from_triples(flat_found_triples):
    context_str = "\n".join([', '.join(triple[:3]) for triple in flat_found_triples])
    if len(context_str) > SYMBOL_COUNT_LIMIT:
        print(f"TOO LONG CONTEXT - {len(context_str)}. Shortening in half")
        return get_context_str_from_triples(flat_found_triples[:len(flat_found_triples)//2])
    else:
        return context_str


def get_answers_questions_samples(llm, questions_data: list, graph: Graph):
    # named_entity_json = {"named_entities": []}
    # prompt_constructor = TaskSplitPromptConstructor()
    # prompt = prompt_constructor.get_task_split_prompt(task="triple", split_type='sentence')

    # answers = {}
    # supporting_facts = {}
    questions_answers_samples = []
    for question_info in tqdm(questions_data):
        for entity in ['question', 'query']:
            question = question_info.get(entity)
        if not question:
            raise ValueError(f"{question_info=}")
        # question_id = question_info['id']

        query_ner_messages = ChatPromptTemplate.from_messages([SystemMessage(SYSTEM_PROMPT),
                                                          HumanMessage(ONE_SHOT_INPUT_PROMPT),
                                                          AIMessage(ONE_SHOT_OUTPUT_PROMPT),
                                                          HumanMessage(USER_PROMPT_TEMPLATE.format(question))])
        # prompt = self.task_split_prompt_constructor.get_task_split_prompt(task="entity", split_type=self.split_type)
        # ner_messages = prompt.get_prompt().format_prompt(user_input=passage)
        # query_ner_messages = query_ner_prompts.format_prompt()
        chat_completion = ''
        temperature = 0.0
        for attempt in range(ATTEMPTS):
            try:
                chat_completion = llm.invoke(query_ner_messages, max_tokens=1024, task='ner', temperature=temperature)
                break
            except Exception as err:
                print(f"ERROR: {err}\nTRYING again")
                temperature += 0.2

        response_content = chat_completion[-1]['content']   # .content
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
        # print("\n"+str(query_entities))
        if len(query_entities) == 0 or type(query_entities) != list or type(query_entities[0]) != str:
            short_answer = ""
            flat_found_triples = []
        else:
            query_entities = [processing_phrases(p) for p in query_entities]
            found_triples = graph.search(query_entities)
            flat_found_triples = [triple for triples in found_triples for triple in triples]
            
            context_str = get_context_str_from_triples(flat_found_triples)
            context_str = f'Context:\n{context_str}'
            query_ner_messages = ChatPromptTemplate.from_messages([SystemMessage(SYSTEM_PROMPT_QA),
                                                          HumanMessage(USER_PROMPT_TEMPLATE_QA.format(
                                                              context=context_str, question=question, 
                                                              answer_key_word=KEY_WORD))])
            answer_chat_completion = llm.invoke(query_ner_messages, max_tokens=1536, task='ner')
            answer_str = answer_chat_completion[-1]['content']   # .content
            answer_start_index = answer_str.find(KEY_WORD)
            if answer_start_index != -1:
                short_answer = answer_str[answer_start_index+len(KEY_WORD):]
            else:
                short_answer = ""

        # print(f"{question}\n{context_str=}")

        questions_answers_samples.append({"question": question, "answer": short_answer, "evidence": flat_found_triples})
        # answers[question_id] = short_answer
        # supporting_facts[question_id] = flat_found_triples
        
    return questions_answers_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=Path, required=True)
    parser.add_argument('--llm_path', type=Path, required=True)
    parser.add_argument('--questions_path', type=Path, required=True)
    parser.add_argument('--benchmark_name', type=str, choices=['hotPotQA', 'multiHopRAG'], required=True)

    args = parser.parse_args()
    working_dir = args.working_dir
    llm_path = args.llm_path
    questions_path = args.questions_path
    benchmark_name = args.benchmark_name

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

    predictions = get_answers_questions_samples(llm, questions_data, graph)

    output_file_path = os.path.join(Path(working_dir), Path(f"{benchmark_name}_raw_preds.json"))
    with open(output_file_path, mode='w') as output_file:
        json.dump(predictions, output_file)


if __name__ == '__main__':
    main()
