from component import Component
from pathlib import Path
import json
from tqdm import tqdm

from models_utils.embedding_model import EmbeddingModel


class TextsEmbedder(Component):
    def __init__(self, component_name: str, log: bool, working_dir: Path, 
                 embedding_model_path: Path, batch_size: int,
                 input_file: Path, output_file: str, seed: int) -> None:
        super().__init__(component_name, log, working_dir)
        self.input_file_path = Path(input_file)
        self.embedding_model = EmbeddingModel(embedding_model_path, batch_size)
        self.batch_size = batch_size
        self.seed = seed
        self.output_file = output_file

    def read_raw_triples_entities(self):
        reading_path = self.working_dir / self.input_file_path
        with open(reading_path) as f:
            extracted_entities_triples = json.load(f)

        return extracted_entities_triples

    def add_embeddings_to_texts(self, extracted_entities_triples):
        batched_texts = [[]]
        for index, sample in enumerate(extracted_entities_triples):
            text = sample['text_fragment']
            # text_id = sample.get('idx', index)
            if not sample.get('idx') or sample['idx'] is None:
                text_id = index
            text_sample = {'text': text, 'idx': text_id}
            if len(batched_texts[-1]) == self.batch_size:
                batched_texts.append([text_sample])
            else:
                batched_texts[-1].append(text_sample)
        
        id_embeddings_list = []
        for text_batch in tqdm(batched_texts):
            texts = [sample['text'] for sample in text_batch]
            embeddings = self.embedding_model(texts)
            # assert len(embeddings) == self.batch_size, f"{len(embeddings)=} | {self.batch_size=}"
            for embedding, id_text_dict in zip(embeddings, text_batch):
                id_embedding_dict = {"id": id_text_dict['idx'], "embeddings": embedding.tolist()}
                id_embeddings_list.append(id_embedding_dict)

        return id_embeddings_list, []

    def __call__(self) -> None:
        extracted_entities_triples = self.read_raw_triples_entities()

        text_embeddings, stats = self.add_embeddings_to_texts(extracted_entities_triples)
        self.write_statistics(stats)
        self.write_result(text_embeddings)

    def write_result(self, ids_embeddings_list):
        writing_path = self.working_dir / Path(self.output_file)
        with open(writing_path, 'w') as file:
            json.dump(ids_embeddings_list, file, indent=4)
        print('Processed passages saved to', self.working_dir / self.output_file)