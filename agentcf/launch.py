import random
random.seed = 51

import argparse
from utils import read_amazon_data, read_prompts, preprocess_data, prepare_train_test_set, write_json_file, read_json_file
from agents_system import AgentCFSystem
from model import LLM, AI_Model
from tests import check_train_test_set_integrity
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--use_gpt35', action='store_true')
    parser.add_argument('--use_mistral', action='store_true')

    parser.add_argument('--prompts_path', type=str, required=True)
    parser.add_argument('--path_to_data', type=str, required=True)
    parser.add_argument('--path_to_init_user_profiles', type=str)
    parser.add_argument('--path_to_init_item_profiles', type=str)
    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--dataset_type', choices=['amazon_cds', 'movie_lens'], type=str, required=True)
    parser.add_argument('--max_num_of_steps', type=int)
    parser.add_argument('--max_num_of_substeps', type=int, default=2)
    parser.add_argument('--popular_negative', action='store_true')
    parser.add_argument('--random_neg_item_sampling', action='store_true', help='if true, \
                        during test phase negative examples will be sampled randomly just to \
                        be not connected with user, rather than be examples with negative overall for this user')
    
    # parser.add_argument('--write_each_num_steps', type=int, default=20)
    parser.add_argument('--save_each_num_steps', type=int, default=20)
    
    arguments = parser.parse_args()

    # Prepare data
    data = read_amazon_data(arguments.path_to_data)
    all_prompts = read_prompts(arguments.prompts_path)
    users, items, records = preprocess_data(data, arguments.dataset_type)
    positive_records, negative_records_users, neutral_records_users, positive_test_records = prepare_train_test_set(records, 
                                                                                        arguments.popular_negative)
    check_train_test_set_integrity(users, positive_records, positive_test_records, 
                                   neutral_records_users, negative_records_users)
    print('Data prepared')
    
    # Prepare model
    if arguments.use_gpt35:
        openai_key = os.getenv("OPENAI_KEY")
        if openai_key:
            llm = AI_Model(model_type='gpt35', key=openai_key)
        else:
            raise ValueError("if you want to use openai GPT model set OPENAI_KEY env variable")
    elif arguments.use_mistral:
        mistralai_key = os.getenv("MISTRAL_KEY")
        if mistralai_key:
            llm = AI_Model(model_type='mistral', key=mistralai_key)
        else:
            # llm = lambda x: x.replace('fo', 'pld')
            raise ValueError("if you want to use openai Mistral model set MISTRAL_KEY env variable")    
    elif arguments.model_path:
        llm = LLM(arguments.model_path)
    else:
        raise ValueError("if you don't want to use openai GPT model, set --model_path")
    print('Model ready')

    # Creating system
    save_each_num_steps = arguments.save_each_num_steps

    init_users_profiles = None
    init_items_profiles = None
    num_begin_step = 0
    if arguments.path_to_init_user_profiles and arguments.path_to_init_item_profiles:
        init_users_profiles = read_json_file(arguments.path_to_init_user_profiles)
        init_items_profiles = read_json_file(arguments.path_to_init_item_profiles)
        num_begin_step = init_users_profiles['step']
    else:
        print("There is no user or items profiles, so start from beginning")
    agent_system = AgentCFSystem(users, items, llm, all_prompts['init'], arguments.dataset_type,
                                 init_users_profiles, init_items_profiles, arguments.output_path)
    print("Agent system prepared\nTraining..")


    if arguments.max_num_of_steps:
        num_iterations = min(len(positive_records) - num_begin_step, arguments.max_num_of_steps)
    else:
        num_iterations = len(positive_records)
    # Train system
    # test_results_before_training = agent_system.test_agents(negative_records_users, neutral_records_users, positive_records, 
    #                                         positive_test_records, all_prompts['forward']['user_ranking'], 
    #                                         arguments.random_neg_item_sampling)
    
    # write_json_file(test_results_before_training, 'test_before_train', arguments.output_path)
    all_steps_profiles = agent_system.train_agents_memory(positive_records, negative_records_users, neutral_records_users, 
                                                          all_prompts, num_begin_step,
                                     num_iterations, arguments.max_num_of_substeps, save_each_num_steps)
    print(f"\nFINISHED_TRAIN {'-'*20}\n")
    test_results_after_training = agent_system.test_agents(negative_records_users, neutral_records_users, positive_records, 
                                            positive_test_records, all_prompts['forward']['user_ranking'],
                                            arguments.random_neg_item_sampling)
    print(f"\nFINISHED_TEST {'-'*20}\n")
    
    write_json_file(all_steps_profiles, 
                    f'UserProfiles_{num_iterations}Steps_{save_each_num_steps}WriteFrequency', arguments.output_path)
    write_json_file(test_results_after_training, 'test_after_train', arguments.output_path)
    print("END!")

if __name__ == '__main__':
    main()
