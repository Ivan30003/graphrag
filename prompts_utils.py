from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from abc import ABC, abstractmethod


ONE_SHOT_PARAGRAPH = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.
"""

ONE_SHOT_PARAGRAPH_ENTITIES = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

ONE_SHOT_PARAGRAPH_TRIPLES = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"]
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

ONE_SHOT_PARAGRAPH_INPUT = """Paragraph:
```
{}
```
""".format(ONE_SHOT_PARAGRAPH)

NER_SYSTEM_INSTRUCTION_PARAGRAPH = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

NER_USER_INPUT_PARAGRAPH = "Paragraph:```\n{user_input}\n```"


OPENIE_SYSTEM_INSTRUCTION_PARAGRAPH = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

OPENIE_INSTRUCTION_PARAGRAPH_TEMPLATE = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""

OPENIE_ONE_SHOT_PARAGRAPH_INPUT = OPENIE_INSTRUCTION_PARAGRAPH_TEMPLATE.replace("{passage}", ONE_SHOT_PARAGRAPH).replace("{named_entity_json}", ONE_SHOT_PARAGRAPH_ENTITIES)




## SENTENCE
## SENTENCE
## SENTENCE




ONE_SHOT_SENTENCE = """Radio City is India's first private FM radio station and was started on 3 July 2001"""

ONE_SHOT_SENTENCE_ENTITIES = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "private FM radio station"]
}
"""

ONE_SHOT_SENTENCE_TRIPLES = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"]
    ]
}
"""

ONE_SHOT_SENTENCE_INPUT = "Sentence:\n```\n{}\n```".format(ONE_SHOT_SENTENCE)

NER_USER_INPUT_TEMPLATE_SENTENCE = "Sentence:\n```\n{user_input}\n```"

NER_SYSTEM_INSTRUCTION_SENTENCE = """Your task is to extract named entities from the given sentence. 
Respond with a JSON list of named entities.
"""

OPENIE_SYSTEM_INSTRUCTION_SENTENCE = """Your task is to construct an RDF (Resource Description Framework) graph from the given sentence and named entity list. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each sentence.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

OPENIE_INSTRUCTION_SENTENCE_TEMPLATE = """Convert the sentence into a JSON dict, it has a named entity list and a triple list.
Sentence:
```
{passage}
```

{named_entity_json}
"""

OPENIE_ONE_SHOT_SENTENCE_INPUT = OPENIE_INSTRUCTION_SENTENCE_TEMPLATE.replace("{passage}", ONE_SHOT_SENTENCE).replace("{named_entity_json}", ONE_SHOT_SENTENCE_ENTITIES)


class OneShotPrompt(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_prompt(self):
        pass


class OneShotPromptParagraphEntities(OneShotPrompt):
    def __init__(self) -> None:
        self.ner_system_instruction = NER_SYSTEM_INSTRUCTION_PARAGRAPH
        self.ner_input_one_shot = ONE_SHOT_PARAGRAPH_INPUT
        self.ner_output_one_shot = ONE_SHOT_PARAGRAPH_ENTITIES
        self.ner_user_input = NER_USER_INPUT_PARAGRAPH

    def get_prompt(self):
        return ChatPromptTemplate.from_messages([SystemMessage(self.ner_system_instruction),
                                                        HumanMessage(self.ner_input_one_shot),
                                                        AIMessage(self.ner_output_one_shot),
                                                        HumanMessagePromptTemplate.from_template(self.ner_user_input)])


class OneShotPromptParagraphTriples(OneShotPrompt):
    def __init__(self) -> None:
        self.openie_post_ner_instruction = OPENIE_SYSTEM_INSTRUCTION_PARAGRAPH
        self.openie_post_ner_input_one_shot = OPENIE_ONE_SHOT_PARAGRAPH_INPUT
        self.openie_post_ner_output_one_shot = ONE_SHOT_PARAGRAPH_TRIPLES
        self.openie_post_ner_frame = OPENIE_INSTRUCTION_PARAGRAPH_TEMPLATE

    def get_prompt(self):
        return ChatPromptTemplate.from_messages([SystemMessage(self.openie_post_ner_instruction),
                                                 HumanMessage(self.openie_post_ner_input_one_shot),
                                                 AIMessage(self.openie_post_ner_output_one_shot),
                                                 HumanMessagePromptTemplate.from_template(self.openie_post_ner_frame)])
        

class OneShotPromptSentenceEntities(OneShotPrompt):
    def __init__(self) -> None:
        self.ner_system_instruction = NER_SYSTEM_INSTRUCTION_SENTENCE
        self.ner_input_one_shot = ONE_SHOT_SENTENCE_INPUT
        self.ner_output_one_shot = ONE_SHOT_SENTENCE_ENTITIES
        self.ner_user_input = NER_USER_INPUT_TEMPLATE_SENTENCE

    def get_prompt(self):
        return ChatPromptTemplate.from_messages([SystemMessage(self.ner_system_instruction),
                                                 HumanMessage(self.ner_input_one_shot),
                                                 AIMessage(self.ner_output_one_shot),
                                                 HumanMessagePromptTemplate.from_template(self.ner_user_input)])


class OneShotPromptSentenceTriples(OneShotPrompt):
    def __init__(self) -> None:
        self.openie_system_instruction = OPENIE_SYSTEM_INSTRUCTION_SENTENCE
        self.openie_post_ner_input_one_shot = OPENIE_ONE_SHOT_SENTENCE_INPUT
        self.openie_post_ner_output_one_shot = ONE_SHOT_SENTENCE_TRIPLES
        self.openie_post_ner_frame = OPENIE_INSTRUCTION_SENTENCE_TEMPLATE

    def get_prompt(self):
        return ChatPromptTemplate.from_messages([SystemMessage(self.openie_system_instruction),
                                                 HumanMessage(self.openie_post_ner_input_one_shot),
                                                 AIMessage(self.openie_post_ner_output_one_shot),
                                                 HumanMessagePromptTemplate.from_template(self.openie_post_ner_frame)])
        

class TaskSplitPromptConstructor:
    def __init__(self) -> None:
        self.one_shot_paragraph_entity_prompt = OneShotPromptParagraphEntities()
        self.one_shot_paragraph_triple_prompt = OneShotPromptParagraphTriples()
        self.one_shot_sentence_entity_prompt = OneShotPromptSentenceEntities()
        self.one_shot_sentence_triple_prompt = OneShotPromptSentenceTriples()

    def get_task_split_prompt(self, task, split_type):
        if task == "entity" and split_type == "paragraph":
            return self.one_shot_paragraph_entity_prompt
        elif task == "triple" and split_type == "paragraph":
            return self.one_shot_paragraph_triple_prompt
        elif task == "entity" and split_type == "sentence":
            return self.one_shot_sentence_entity_prompt
        elif task == "triple" and split_type == "sentence":
            return self.one_shot_sentence_triple_prompt
        else:
            raise ValueError(f"{task=}\n{split_type=}")
