import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from state import state
import argparse
from pathlib import Path
import shutil
import os
import re
import json
import time
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from files_utils import read_yaml_file
from pipeline import Pipeline
from graph_utils.graph_process import GraphConstructor

TXT_INPUT_TEMPLATE = """
Your text file should contain only info paragraphs divided with \\n\\n. Like this:
aaaaaaaaaabbbbbb

cccccccddddddddd

wwwwwwwwwwwwwwww
"""

STATE = {}


# query_prompt_one_shot_input = """Please extract all named entities from provided sentence.
# Do not answer with entities, which are not present in sentence
# Place the named entities in json, containing list of extracted entities.

# Sentence: 
# ```
# Which magazine was started first Arthur's Magazine or First for Women
# ```
# """
# query_prompt_one_shot_output = """
# {"named_entities": ["magazine", "woman", "First for Women", "Arthur's Magazine"]}
# """

# query_prompt_template = """
# Sentence: 
# ```
# {}
# ```
# """


def process_query(prompt):
    query_entities = extract_query_entities(prompt.strip('\n').replace('?',''))
    print(f"{query_entities=}")
    

def extract_doc_info(doc_save_path):
    pipeline = STATE['pipeline']
    dataset_processor = pipeline.components[0]
    dataset_processor.dataset_path = Path(doc_save_path)
    pipeline.launch()


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


def extract_query_entities(query):
    llm = STATE['llm']

    prompt_constructor = STATE['pipeline'].components[1].task_split_prompt_constructor
    prompt = prompt_constructor.get_task_split_prompt(task="triple", split_type='sentence')
    ner_messages = prompt.get_prompt().format_prompt(user_input=query, )

    chat_completion = llm.invoke(ner_messages.to_messages(), temperature=0)
    response_content = chat_completion[4]['content']   # .content
    response_content = extract_json_dict(response_content)

    if 'named_entities' not in response_content:
        response_content = []
    else:
        response_content = response_content['named_entities']
    
    # query_ner_prompts = ChatPromptTemplate.from_messages([SystemMessage("You're a very effective entity extraction system."),
    #                                                       HumanMessage(query_prompt_one_shot_input),
    #                                                       AIMessage(query_prompt_one_shot_output),
    #                                                       HumanMessage(query_prompt_template.format(query))])
    # query_ner_messages = query_ner_prompts.format_prompt()
    # # user_prompt = 'Question: ' + query + '\nThought: '
    # chat_completion = llm.invoke(query_ner_messages.to_messages(), temperature=0)
    # response_content = chat_completion[4]['content']   # .content
    # response_content = extract_json_dict(response_content)
    response_content = str(response_content)
    query_ner_list = eval(response_content)['named_entities']
    query_ner_list = [processing_phrases(p) for p in query_ner_list]

    return query_ner_list


def preprocess_file(text):
    start_time = time.time()
    save_dir = STATE['working_dir']
    graph_constructor = STATE['graph_constructor']

    save_path = save_dir + f"saved.txt"
    processed_text_fragments = [text_fragment.replace('\n','').strip('.').strip(' ') for 
                                text_fragment in text.split('\n\n')]
    with open(save_path, mode='w') as output_file:
        for text_fragment in processed_text_fragments:
            output_file.write(text_fragment)
            output_file.write('\n')
    extract_doc_info(save_path)
    STATE['graph'] = graph_constructor()
    end_time = time.time()
    return f"Processing complete in {round(end_time-start_time, 2)} s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=Path, required=True)

    args = parser.parse_args()
    config_path = args.config_path
    config = read_yaml_file(config_path)
    working_dir = config['working_dir']
    STATE['working_dir'] = working_dir
    print(f"Pipeline initialization...")
    STATE['pipeline'] = Pipeline(config)
    input_file_name = STATE['pipeline'].components[-1].output_files[1]
    STATE['graph_constructor'] = GraphConstructor("graph_constructor", True, working_dir, input_file_name)
    STATE['llm'] = STATE['pipeline'].components[1].llm
    print(f"Pipeline READY")
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                input_text_tab = gr.Textbox(interactive=True, label='place your text here')
                launch_button = gr.Button(value='submit')
                # upload_button = gr.UploadButton(label="upload txt file")
                progress_tab = gr.Textbox(label= 'Progress:', interactive=False)
                pipeline_settings = gr.Dropdown(label='Mode', value='generate', 
                                            choices=['generate', 'upscale', 'change', 'refine'])
                info_box = gr.Textbox(label= 'Hint:', interactive=False, value=TXT_INPUT_TEMPLATE)
                # input_text_file.change(fn=preprocess_file, inputs=[input_text_file], outputs=[progress_tab])
            with gr.Column(scale=1):
                query_box = gr.Textbox(label= 'query', interactive=True, placeholder='input your query here:')
                graph_options = gr.Dropdown(label='Mode', value='generate', interactive=True,
                                            choices=['generate', 'upscale', 'change', 'refine'])
                submit_button = gr.Button(value='submit')
                output_textbox = gr.Textbox(label='Answer', interactive=False, value='')
        # upload_button.upload(fn=preprocess_file, inputs=[input_text_file], outputs=[progress_tab])
        launch_button.click(lambda: tuple([gr.update(interactive=False)] * 3), [],
                               [query_box, pipeline_settings, input_text_tab]).\
        then(fn=preprocess_file, inputs=[input_text_tab], outputs=[progress_tab]).\
        then(lambda: tuple([gr.update(interactive=True)] * 3), [], [query_box, pipeline_settings, input_text_tab])
        # input_text_file.preprocess(fn=preprocess_file, inputs=[input_text_file], outputs=[])
        submit_button.click(fn=process_query, inputs=[query_box], outputs=[output_textbox])

    demo.queue().launch(share=True, server_name='192.168.0.11', server_port=5361)


if __name__ == "__main__":
    main()
