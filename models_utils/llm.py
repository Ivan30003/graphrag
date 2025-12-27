import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import set_seed


MIN_JSON_LENGTH = 5


def init_langchain_model(llm: str, model_name: str, temperature: float = 0.0, 
                         max_retries=5, timeout=60, seed=52, **kwargs):
    """
    Initialize a language model from the langchain library.
    :param llm: The LLM to use, e.g., 'openai', 'together'
    :param model_name: The model name to use, e.g., 'gpt-3.5-turbo'
    """
    set_seed(seed)
    if llm == 'openai':
        # https://python.langchain.com/v0.1/docs/integrations/chat/openai/
        assert model_name.startswith('gpt-')
        return ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model=model_name, temperature=temperature, max_retries=max_retries, timeout=timeout, **kwargs)
    else:
        # add any LLMs you want to use here using LangChain
        # raise NotImplementedError(f"LLM '{llm}' not implemented yet.")
        print(f"{model_name=}")
        if 'phi' in str(model_name).lower():
            model = LLM_Phi_35(model_name)
        elif 't5' in str(model_name).lower():
            model = LLM_T5(model_name)
        elif 'qwen' in str(model_name).lower():
            model = LLM_Qwen_3(model_name)
        elif 'granite' in str(model_name).lower():
            model = LLM_Granite_33(model_name)
        else:
            raise NotImplementedError()
        return model


class LLM_Phi_35:
    def __init__(self, model_name_or_path) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,  # /media/hdd/models/text/Phi-3.5-mini-instruct/
            torch_dtype=torch.float16, 
            trust_remote_code=False,
            attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": ""},
        ]

        self.pipe = pipeline(
            "text-generation",
            model=model,
            device="cuda",
            tokenizer=tokenizer,
        )

    def invoke(self, openie_messages: ChatPromptTemplate, temperature=0.0, max_tokens=3096, task='ner'):
        system_message = openie_messages[0].content
        human_message = openie_messages[1].content
        ai_message = openie_messages[2].content
        message = openie_messages[3].content

        self.messages[0]['content'] = system_message
        self.messages[1]['content'] = human_message
        self.messages[2]['content'] = ai_message
        self.messages[3]['content'] = message

        # print(f"\n\n{'*'*50}\nSystem: {system_message}\nUser: {human_message}\nAI: {ai_message}\nUser: {message}")
        # prompt = self.pipe.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        if temperature == 0.0:
            output = self.pipe(self.messages, max_new_tokens=max_tokens)
        else:
            output = self.pipe(self.messages, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
        # print(f"ANSWER:\n{output[0]['generated_text'][4]['content']}")
        return output[0]['generated_text']
    

class LLM_Qwen_3:
    def __init__(self, model_name_or_path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.max_attempts = 2

        self.messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": ""},
        ]

    def is_repeat(self, text: str):
        phrases = text.split('\n')
        if len(phrases) < MIN_JSON_LENGTH:
            return False
        for i in range(len(phrases)-2):
            if phrases[i] == phrases[i+1] and phrases[i+1] == phrases[i+2]:
                return True

        return False


    def invoke(self, openie_messages: ChatPromptTemplate, temperature=0.0, max_tokens=4096, task='ner'):
        messages_length = len(openie_messages)
        assert messages_length in [2, 4]
        print(f"{'*'*70}\n{openie_messages}\n\n")
        # messages_template = self.messages[:messages_length]
        for ind in range(messages_length):
            self.messages[ind]['content'] = openie_messages[ind].content

        text = self.tokenizer.apply_chat_template(
            self.messages[:messages_length],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        for i in range(self.max_attempts):
            # do_sample = False
            # if temperature > 0.0:
            #     do_sample = True
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            if self.is_repeat(content):
                print(f"ANSWER:\n{[content]}")
                print(f"REPEAT, trying again")
            else:
                break

        print(f"ANSWER:\n{[content]}\n\n\n")
        return [ None, None, None, None, {'content': content}]


class LLM_Granite_33:
    def __init__(self, model_name_or_path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        self.messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": ""},
        ]

        # self.pipe = pipeline(
        #     "text-generation",
        #     model=model,
        #     device="cuda",
        #     tokenizer=tokenizer,
        # )

    def invoke(self, openie_messages: ChatPromptTemplate, temperature=0.0, max_tokens=3096, task='ner'):
        messages_length = len(openie_messages)
        assert messages_length in [2, 4]
        print(f"{'*'*70}\n{openie_messages}\n\n")
        # messages_template = self.messages[:messages_length]
        for ind in range(messages_length):
            self.messages[ind]['content'] = openie_messages[ind].content

        model_inputs = self.tokenizer.apply_chat_template(
            self.messages[:messages_length],
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
            return_dict=True
        ).to(self.model.device)
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        do_sample = False
        if temperature > 0.0:
            do_sample = True
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature
        )
        # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # try:
        #     # rindex finding 151668 (</think>)
        #     index = len(output_ids) - output_ids[::-1].index(151668)
        # except ValueError:
        #     index = 0

        # content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(generated_ids[0, model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # print(f"ANSWER:\n{output[0]['generated_text'][4]['content']}")
        print(f"{'*'*70}\n{content}\n\n")
        return [ None, None, None, None, {'content': content}]


class LLM_T5:
    def __init__(self, model_name_or_path) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, 
                                                   device_map="auto", torch_dtype=torch.float16)
    
    def post_process_triples(self, text):
        return text

    def invoke(self, openie_messages: ChatPromptTemplate, max_tokens=1024, temperature=0.0, task="ner"):  # , max_tokens=3096
        input_text = f"{openie_messages[3].content}"
        if task == "ner":
            input_text = f"Your task is to extract named entities from the given sentence. {openie_messages[3].content}"
        elif task == "openie":
            inputs_list = openie_messages[3].content.split('\n')[1:-2]
            user_input = '\n'.join(inputs_list).strip('\n')
            input_text = f"Your task is to construct an RDF (Resource Description Framework) triples from the given sentence {user_input}"
        print(f"{input_text=}")
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids, max_new_tokens=max_tokens)
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if task == "ner":
            entities = ', '.join([f"\"{entity}\"" for entity in output_text.split(',')])
            output_text = "{\"named_entities\": [" + entities + "]}"
        elif task == "openie":
            output_text = self.post_process_triples(output_text)
            output_text = "{\"triples\": [" + output_text.replace('\'','\"') + "]}"
        print(output_text)
        return [ None, None, None, None, {'content': output_text}]
