from abc import ABC, abstractmethod


class AgentCF(ABC):
    def __init__(self, prompt_template, llm) -> None:
        self.prompt_template = prompt_template
        self.llm = llm

    def get_answer(self, prompt, temperature):
        return self.llm(prompt, temperature)

    @abstractmethod
    def update_memory(self):
        pass

class AgentUser(AgentCF):
    def __init__(self, init_memory, llm) -> None:
        super().__init__(init_memory, llm)
        if type(init_memory) == dict:
            self.short_memory = init_memory['memory'] # self.prompt_template
            self.long_term_memory = [self.short_memory]
            self.preferences = init_memory['general_preferences']
        
    def update_memory(self, new_short_memory):
        self.short_memory = new_short_memory
        self.long_term_memory.append(new_short_memory)

    def update_preferences(self, new_preferences_text):
        self.preferences = new_preferences_text


class AgentItem(AgentCF):
    def __init__(self, prompt_template, llm) -> None:
        super().__init__(prompt_template, llm)
        self.memory = self.prompt_template
        
    def update_memory(self, updated_memory):
        self.memory = updated_memory
