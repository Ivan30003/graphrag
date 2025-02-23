import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from state import state
import argparse
from pathlib import Path
import shutil
import os

from files_utils import read_yaml_file
from pipeline import Pipeline

TXT_INPUT_TEMPLATE = """
Your text file should contain only info paragraphs divided with \\n\\n. Like this:
aaaaaaaaaabbbbbb

cccccccddddddddd

wwwwwwwwwwwwwwww
"""

STATE = {}

def init_pipeline():
    pass
    # torch.set_default_device("cuda")
    # model = AutoModelForCausalLM.from_pretrained("/media/hdd/models/text/phi-2/", torch_dtype="auto", trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained("/media/hdd/models/text/phi-2/", trust_remote_code=True)
    # state['model'] = model
    # state['tokenizer'] = tokenizer


def process_query(prompt):
    # text = '''def print_numbers(list_of_numbers):
    # """
    # return list of only even numbers from the given list
    # """'''
    tokenizer = state['tokenizer']
    model = state['model']

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to('cuda')

    outputs = model.generate(**inputs, max_length=300)
    text = tokenizer.batch_decode(outputs)[0]
    return text
    

# def process_file(fileobj):
#     save_dir = STATE['working_dir']
#     path = "/home/ubuntu/temps/" + os.path.basename(fileobj)
#     shutil.copyfile(fileobj.name, save_dir)
#     # now you can process the file at path as needed, e.g:
#     # do_something_to_file(path)
def extract_doc_info(doc_save_path):
    pipeline = STATE['pipeline']
    dataset_processor = pipeline.components[0]
    dataset_processor.dataset_path = Path(doc_save_path)
    pipeline.launch()
    # triples_file_name = pipeline.parts[].output_files[1]
    return triples_file_name


def constract_graph(graph):
    pass


def preprocess_file(text):
    # state['input_image'] = input_image
    save_dir = STATE['working_dir']
    save_path = save_dir + f"saved.txt"
    processed_text_fragments = [text_fragment.replace('\n','').strip('.').strip(' ') for 
                                text_fragment in text.split('\n\n')]
    with open(save_path, mode='w') as output_file:
        for text_fragment in processed_text_fragments:
            output_file.write(text_fragment)
            output_file.write('\n')
    # shutil.copyfile(file.name, save_path)
    extract_doc_info(save_path)
    # graph = constract_graph()

    return "Complete"


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
