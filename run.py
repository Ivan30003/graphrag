import argparse
from pathlib import Path

# from processing_utils import (dataset_process, base_process_entities_triples, 
# add_embeddings_to_triples_linkage, merging_linkages)
# from triples_entities_extraction import extract_openie
# from llm import init_langchain_model
# from prompts_utils import TaskSplitPromptConstructor
from files_utils import read_yaml_file
# from embedding_model import EmbeddingModel
from pipeline import Pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_path', type=Path)
    parser.add_argument('--config_path', type=Path, required=True)

    args = parser.parse_args()

    # dataset_path = args.dataset_path
    config_path = args.config_path
    # num_passages = args.num_passages
    # model_name_or_path = args.model_name
    # splitting_type = args.split_type
    # output_path = args.output_path
    # dataset_name = dataset_path.split('/')[-1].split('.')[-2] + '_'
    # arg_str = dataset_path.stem
    print("START")
    config = read_yaml_file(config_path)
    pipeline = Pipeline(config)
    pipeline.launch()

    # retrieval_corpus = dataset_process(dataset_path, splitting_type, dataset_name, num_passages)
    # print("CORPUS LOADED")
    # client = init_langchain_model(args.llm, model_name_or_path)  # LangChain model
    # if len(args.embedding_model_path) > 0:
    #     embedding_model = EmbeddingModel(args.embedding_model_path)
    # print("MODELS LOADED")
    # task_split_prompt_constructor = TaskSplitPromptConstructor()
    
    # extracted_entities_triples = extract_openie(retrieval_corpus, client, splitting_type, 
    #                                                      task_split_prompt_constructor)

    # print("EXTRACTED ROW ENTITIES AND TRIPLES")
    # all_entities, all_triples = base_process_entities_triples(extracted_entities_triples)
    # print("PROCESSED ENTITIES AND TRIPLES")
    # all_triples_linkage_embeddings = add_embeddings_to_triples_linkage(all_triples, embedding_model)
    # print("ADDED EMBEDDINGS TO TRIPLES LINKAGE")
    # assert len(all_triples_linkage_embeddings) == len(all_triples)
    # print(f"{all_triples=}")
    # # print(f"{all_triples_linkage_embeddings=}")
    # triples_with_linkage_embeddings = [triple + [vector] for triple, vector in 
    #                                    zip(all_triples, all_triples_linkage_embeddings)]
    # write_csv_file(output_path, triples_with_linkage_embeddings, f"{dataset_name}_triples_and_embeddings.csv")

    # merging_linkages(triples_with_linkage_embeddings)
    
    

    # extra_info_json = {"all_entities": processed_entities,
    #                     "all_triples": processed_triples,
    #                     # "avg_ent_chars": avg_ent_chars,
    #                     # "avg_ent_words": avg_ent_words,
    #                     }

    # output_json_path = f'{output_path}openie_results_{arg_str}.json'
    write_json_file(extra_info_json, output_json_path)
    # avg_ent_chars = np.mean([len(e) for e in all_entities])
    # avg_ent_words = np.mean([len(e.split()) for e in all_entities])
