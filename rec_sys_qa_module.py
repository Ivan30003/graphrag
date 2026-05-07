import argparse
import json
import re
import os
import csv
from pathlib import Path
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from ast import literal_eval
from random import shuffle

from models_utils.llm import LLM_Phi_35, LLM_Qwen_3, LLM_T5
from graph_utils.graph_process import GraphConstructor, Graph
from extractor_utils.triples_entities_extraction import Extractor


SYMBOL_COUNT_LIMIT = 12000
ATTEMPTS = 3
KEY_WORD = "So the answer is:"

# SYSTEM_PROMPT = "You're a very effective entity extraction system."
# ONE_SHOT_INPUT_PROMPT = """Please extract all named entities that are important for solving the questions below.
# Place the named entities in json format.

# Question: Which magazine was started first Arthur's Magazine or First for Women?

# """
# ONE_SHOT_OUTPUT_PROMPT = """
# {"named_entities": ["First for Women", "Arthur's Magazine"]}
# """

# USER_PROMPT_TEMPLATE = """
# Question: {}

# """


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# SYSTEM_PROMPT_QA = "You're thorough assistant responding to questions, based on retrieved context"
# USER_PROMPT_TEMPLATE_QA = """
# Goal:
# 1) Provide clear and accurate response: carefully review and verify the retrieved data, and integrate any relevant necessary knowledge to comprehensively address user's question.
# 2) Do not fabricate information: if you are unsure of answer just say so.
# 3) Do not include details, not supported by the provided evidence
# 4) Place your short answer to the question after words "{answer_key_word}"

# Context:
# {context}

# User's question:
# {question}
# """
# KEY_WORD = "So the answer is:"


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


def get_entities_and_items(test_items: dict):
    input_entities = []
    items_properties = []
    for item in test_items:
        item_categories_list = item['category'][1:]
        item_categories_str = ', '.join(item_categories_list)
        item_properties_str = f"Title: {item['title']}; category: {item_categories_str}; brand: {item['brand']}"

        items_properties.append(item_properties_str)
        input_entities.extend([item_category.lower() for item_category in item_categories_list])
        input_entities.append(item['title'].lower())
        input_entities.append(item['brand'].lower())

    items_properties_str = '\n'.join(items_properties)
    return input_entities, items_properties_str


def define_gt(test_items):
    test_items_index = [(ind+1, items) for ind, items in enumerate(test_items)]
    print(f"\n\n{[test_ind for test_ind in test_items_index]=}\n\n")
    test_items_index_sorted = sorted(test_items_index, key=lambda x: x[1]['overall'], reverse=True)
    print(f"\n\n{test_items_index_sorted=}\n\n")
    ground_truth_list = [ind_item[0] for ind_item in test_items_index_sorted]

    return ground_truth_list


def get_answers_questions_samples(llm: LLM_Qwen_3, users_test_items: list, graph: Graph, prompt_template: str, debug):
    questions_answers_samples = []
    for question_info in tqdm(users_test_items):
        test_items = question_info.get('question')
        if not test_items:
            raise ValueError(f"{question_info=}")

        input_entities, items_properties_str = get_entities_and_items(test_items)
        ground_truth = define_gt(test_items)
        
        ### DEBUG
        if debug:
            print(f"First five input entities:\n{input_entities[:5]}\n")
            print(f"First of items properties:\n{items_properties_str[:200]}...\n")
            print(f"GT list:\n{ground_truth}\n")

        if len(input_entities) == 0 or type(input_entities) != list or type(input_entities[0]) != str:
            flat_found_triples = []
            print(f"Haven't found any user info {items_properties_str}")
        else:
            found_triples = graph.search(input_entities)
            flat_found_triples = [triple[:3] for triples in found_triples for triple in triples]
            
            ### DEBUG
            if debug:
                print(f"First five found triples:\n{flat_found_triples[:5]}\n")
            
            context_str = get_context_str_from_triples(flat_found_triples)
            prompt = prompt_template.format(user_triples=context_str, items_information=items_properties_str)
            query_ner_messages = ChatPromptTemplate.from_messages([SystemMessage("You are helpful assistant \
                                                                                 for recommendation systems"),
                                                          HumanMessage(prompt)])
            # context_str = f'Context:\n{context_str}'
            
            try:
                answer_chat_completion = llm.invoke(query_ner_messages, max_tokens=2048, task='ner')
            except Exception as err:
                print(f"ERROR: {err}\nTRYING AGAIN")
                try:
                    answer_chat_completion = llm.invoke(query_ner_messages, max_tokens=1536, task='ner')
                except:
                    print(f"ERROR: {err}\nFAILED")
                    answer_chat_completion = [{"content": ""}]
            ranking_text = answer_chat_completion[-1]['content']   # .content
            
            ### DEBUG
            if debug:
                print(f"LLM ANSWER:\n{ranking_text}")

            predicted_list = simple_prepare_eval_list(ground_truth, ranking_text)

            ### DEBUG
            if debug:
                print(f"PREDICTED LIST:\n{predicted_list}")

            # ground_truth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            # answer_start_index = answer_str.find(KEY_WORD)
            # if answer_start_index != -1:
            #     short_answer = answer_str[answer_start_index+len(KEY_WORD):]
            # else:
            #     short_answer = ""
            

        questions_answers_samples.append({"question": question_info['id'], 
                                          "answer": predicted_list, 
                                          "label": ground_truth})
        
    return questions_answers_samples


def simple_prepare_eval_list(uniting_record_list, ranking_text):
    try:
        start_index = ranking_text.find('[')
        end_index = ranking_text.find(']')
        assert start_index > -1 and end_index > -1, 'not found [ and ]'
        cutted_text = ranking_text[start_index: end_index+1]
        eval_list = literal_eval(cutted_text)
        assert type(eval_list) == list, 'not list'
        assert len(eval_list) == 10, 'length is not 10'
    except Exception as err:
        print(f"error: {err} in {ranking_text}")
        
    return eval_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=Path, required=True)
    parser.add_argument('--llm_path', type=Path, required=True)
    parser.add_argument('--test_items_path', type=Path, required=True)
    parser.add_argument('--use_enriched_triples', action="store_true")
    parser.add_argument('--benchmark_name', type=str, choices=['cds_vinyl'], required=True)
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    working_dir = args.working_dir
    llm_path = args.llm_path
    test_items_path = args.test_items_path
    benchmark_name = args.benchmark_name
    use_enriched_triples = args.use_enriched_triples
    debug = args.debug

    with open(f"prompts/rank_items.txt") as input_file:
        prompt_template = input_file.read()

    with open(test_items_path, mode='r') as input_file:
        test_items = json.load(input_file)

    if use_enriched_triples:
        triples_file_path = os.path.join(Path(working_dir), Path("enriched_triples.csv"))
    else:
        triples_file_path = os.path.join(Path(working_dir), Path("all_triples.csv"))
    triples = []
    with open(triples_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            for ind in range(len(row)):
                row[ind] = row[ind].lower()
            triples.append(row)
    
    ### DEBUG
    if debug:
        print(f"LLM: {Path(llm_path).name}\n")
    
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

    ### DEBUG
    if debug:
        print(f"Graph dict total length: {len(graph.triples_dict)}")
        print(f"First element: {list(graph.triples_dict.keys())[0]}:\n \
              {graph.triples_dict[list(graph.triples_dict.keys())[0]]}\n\n")
    
    predictions = get_answers_questions_samples(llm, test_items, graph, prompt_template, debug)
    llm_name = str(Path(llm_path).stem)
    if use_enriched_triples:
        output_file_name = Path(f"{benchmark_name}_{llm_name}_with_HM_raw_preds.json")
    else:
        output_file_name = Path(f"{benchmark_name}_{llm_name}_raw_preds.json")
    output_file_path = os.path.join(Path(working_dir), output_file_name)
    with open(output_file_path, mode='w') as output_file:
        json.dump(predictions, output_file)


if __name__ == '__main__':
    main()
