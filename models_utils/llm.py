import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_openai import ChatOpenAI


def init_langchain_model(llm: str, model_name: str, temperature: float = 0.0, max_retries=5, timeout=60, **kwargs):
    """
    Initialize a language model from the langchain library.
    :param llm: The LLM to use, e.g., 'openai', 'together'
    :param model_name: The model name to use, e.g., 'gpt-3.5-turbo'
    """
    if llm == 'openai':
        # https://python.langchain.com/v0.1/docs/integrations/chat/openai/
        assert model_name.startswith('gpt-')
        return ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model=model_name, temperature=temperature, max_retries=max_retries, timeout=timeout, **kwargs)
    else:
        # add any LLMs you want to use here using LangChain
        # raise NotImplementedError(f"LLM '{llm}' not implemented yet.")
        model = LLM(model_name)
        return model


class LLM:
    def __init__(self, model_name_or_path) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,  # /media/hdd/models/text/Phi-3.5-mini-instruct/
            torch_dtype=torch.float16, 
            trust_remote_code=True,
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

        self.generation_args = {
            "max_new_tokens": 1000,
            "return_full_text": False,
            "temperature": 0.0,
        }

    def invoke(self, openie_messages, temperature=0.0, max_tokens=3096):
        system_message = openie_messages[0].content
        human_message = openie_messages[1].content
        ai_message = openie_messages[2].content
        message = openie_messages[3].content

        self.messages[0]['content'] = system_message
        self.messages[1]['content'] = human_message
        self.messages[2]['content'] = ai_message
        self.messages[3]['content'] = message

        print(f"\n\n{'*'*50}\nSystem: {system_message}\nUser: {human_message}\nAI: {ai_message}\nUser: {message}")
        # prompt = self.pipe.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        output = self.pipe(self.messages, max_new_tokens=max_tokens, temperature=temperature)
        print(f"ANSWER:\n{output[0]['generated_text'][4]['content']}")
        return output[0]['generated_text']