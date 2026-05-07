from typing import Any
import torch
from transformers import pipeline, AutoModel
from prompt_templates_for_llm import messages
from openai import OpenAI
from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage

from time import sleep
from openai import RateLimitError 
# import backoff
SLEEP_COOLDOWN = 5


class LLM:
    def __init__(self, model_path) -> None:
        model_rep_name = model_path.lower().split('/')[-1]
        if 'llama' in model_rep_name or 'zephyr' in model_rep_name:
            self.model = pipeline("text-generation", model=model_path, device_map='auto', torch_dtype=torch.bfloat16, return_full_text=False)
            self.prompt_template = messages
        else:
            raise ValueError()
        
    def __call__(self, prompt):
        self.prompt_template[1]['content'] = prompt
        final_prompt = self.model.tokenizer.apply_chat_template(self.prompt_template, tokenize=False, add_generation_prompt=True)
        outputs = self.model(final_prompt, max_new_tokens=320, do_sample=False)  # , temperature=0.7, top_k=5, top_p=0.95
        text = outputs[0]["generated_text"]
        return text
    

class AI_Model:
    def __init__(self, model_type, key: str) -> None:
        if model_type=="gpt35":
            self.model_name = "gpt-3.5-turbo-16k"
            self.client = OpenAI(api_key=key)
            self.get_response_func = self.client.chat.completions.create
        elif model_type=="mistral":
            self.model_name = "mistral-large-latest"
            self.client = MistralClient(api_key=key)
            self.get_response_func = self.client.chat
        
        self.max_tokens = 320
        self.default_temperature = 0.0
        self.max_tries = 5
        self.sum_num_of_used_tokens = 0
        
    def __call__(self, query, temperature=0.0) -> Any:
        print(f"{query}{'-'*30}\n")
        result = ''
        if temperature:
            set_temperature = temperature
        else:
            set_temperature = self.default_temperature
        for i in range(self.max_tries):
            try:
                response = self.get_response_func(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are smart assistant for simulating recommendation system"},
                        {"role": "user", "content": f"{query}"}
                    ], temperature=set_temperature
                )
                result = response.choices[0].message.content
                if 'gpt' in self.model_name:
                    finish_reason = response.choices[0].finish_reason
                    if finish_reason == 'length':
                        print(f"Not get into context length!")
                        return ''
                    elif finish_reason != 'stop':
                        set_temperature = 0.2
                        print(f"trying again because of {finish_reason=}")
                        continue
                        
            except Exception as err:
                if "timed out" in str(err).lower() or type(err) == RateLimitError:
                    print(f"TIMEOUT. Trying again in {SLEEP_COOLDOWN} seconds")
                    sleep(SLEEP_COOLDOWN)
                else:
                    print(f"Error: {err} in getting request")
                    return result
        if response:
            self.sum_num_of_used_tokens += response.usage.prompt_tokens + response.usage.completion_tokens * 3
        else:
            return ''
        # completion tokens are 3 time more expensive
        print(f"{result=}")
        print(f"\nUSED_TOKENS = {self.sum_num_of_used_tokens}")
        print(f">>>>>>>>>>>>>>>>")
        return result


class EmbeddingModel:
    def __init__(self, model_path) -> None:
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to("cuda:0")

    def __call__(self, texts: list) -> Any:
        embeddings = self.model.encode(texts, max_length=2048, batch_size=4)
        return embeddings
