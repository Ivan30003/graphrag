import argparse
import os
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from random import sample
from numpy.linalg import norm

from utils import read_amazon_data, read_prompt_template, extract_text_by_phrase
from agent_cf import AgentItem, AgentUser
from model import LLM, AI_Model

DEBUG = True


def print_input_output(func):
    def func_wrapper(*args, **kwargs):
        if DEBUG:
            print(f"{'-'*30}\n{func.__name__}:\n")
            for arg in args:
                print(f"{arg=}")

        outputs = func(*args, **kwargs)

        if DEBUG:
            print("\n")
            try:
                for output in outputs:
                    print(f"{output=}")
            except:
                print(f"{outputs}")
        
        return outputs
 
    return func_wrapper


def check_answer(pos_item_title, neg_item_title, answer):
    if pos_item_title in answer and neg_item_title not in answer:
        return True
    else:
        return False


def find_some_record(cur_pos_record, pos_records):  # Get record, which is not connected to current user
    cur_user_id = cur_pos_record['reviewer_id']
    for sample in pos_records:
        if sample['reviewer_id'] != cur_user_id:
            return sample


def get_9_additional_records(negative_records_users, neutral_records_users, user_id, record, positive_records):
    neg_records = []
    if negative_records_users.get(user_id):
        neg_records_user_list = negative_records_users[user_id]
        max_sample_num = min(len(neg_records_user_list), 9)
        neg_records.extend(sample(neg_records_user_list, k=max_sample_num))
    
    if neutral_records_users.get(user_id) and len(neg_records) < 9:
        neg_records_user_list = neutral_records_users[user_id]
        max_sample_num = min(9 - len(neg_records), len(neg_records_user_list))
        neg_records.extend(sample(neg_records_user_list, k=max_sample_num))
    
    if len(neg_records) < 9:  # HARDCODE according to the article 
        print(f"Warning! Need additional negative records on this user: {user_id}")
        neg_records.sort(key=lambda x: x['overall'], reverse=True)
        left_records_num = 9 - len(neg_records)
        additional_records = find_some_disconnected_record(record, positive_records, neg_records, left_records_num)
        neg_records.extend(additional_records)


def find_some_disconnected_record(cur_pos_record, pos_records, check_list_records, num):
    cur_user_id = cur_pos_record['reviewer_id']
    additional_records = []
    for record in pos_records:
        if len(additional_records) == num:
            return additional_records

        if record['reviewer_id'] != cur_user_id and record not in check_list_records:
            additional_records.append(record)


def prepare_train_test_set(records, popular_negative, polar_pair=True, items_per_step=2):
    negative_records_users = {}  # user - key
    neutral_records_users = {}  # user - key
    positive_records_users = {}  # user - key
    positive_train_records = []
    positive_test_records = []

    if polar_pair:
        for record in records:
            if record['overall'] < 3.0 and (not popular_negative or record['popularity'] > 30):
                if negative_records_users.get(record['reviewer_id']):
                    negative_records_users[record['reviewer_id']].append(record)
                else:
                    negative_records_users[record['reviewer_id']] = [record]
            elif record['overall'] > 3.0:
                if positive_records_users.get(record['reviewer_id']):
                    positive_records_users[record['reviewer_id']].append(record)
                else:
                    positive_records_users[record['reviewer_id']] = [record]
            elif 2.0 < record['overall'] < 4.0:
                if neutral_records_users.get(record['reviewer_id']):
                    neutral_records_users[record['reviewer_id']].append(record)
                else:
                    neutral_records_users[record['reviewer_id']] = [record]
    else:
        pass

    for user in positive_records_users:
        positive_records_users[user].sort(key=lambda x: x['unix_time'])
        last_positive_item = positive_records_users[user].pop()
        positive_test_records.append(last_positive_item)

    for user in positive_records_users:
        positive_train_records.extend(positive_records_users[user])
    positive_train_records.sort(key=lambda x: x['unix_time'])

    return positive_train_records, negative_records_users, neutral_records_users, positive_test_records


def preprocess_data(data):
    items_ids = []
    items = []
    items_ids_users_pair = []
    users = []
    records = []
    for sample in data:
        if (sample['asin'], sample['reviewer_id']) not in items_ids_users_pair:
            if sample['asin'] not in items_ids:
                items.append({'item_id': sample['asin'], 'item_name': sample['title'], 
                            'item_category': sample['category'], 'overall': sample['overall'], 
                            'popularity': sample['popularity']})
                items_ids.append(sample['asin'])
            records.append(sample)
            items_ids_users_pair.append((sample['asin'], sample['reviewer_id']))
            if sample['reviewer_id'] not in users:
                users.append(sample['reviewer_id'])
        else:
            pass  # print(sample['asin'])

    return users, items, records


def read_prompts(prompts_files_dir):
    all_prompts = {}

    forward_dir = f"{prompts_files_dir}/forward"
    backward_dir = f"{prompts_files_dir}/backward"
    init_dir = f"{prompts_files_dir}/initialization"

    all_prompts['forward'] = {'user': read_prompt_template(f"{forward_dir}/user_agent_prompt.txt"), 
                              'user_ranking': read_prompt_template(f"{forward_dir}/user_agent_test_prompt.txt")}
    all_prompts['backward'] = {'user': read_prompt_template(f"{backward_dir}/user_agent_prompt.txt"), 
                                'item': read_prompt_template(f"{backward_dir}/item_agent_prompt.txt")}
    all_prompts['init'] = {'user': read_prompt_template(f"{init_dir}/user_agent_prompt.txt"), 
                           'item': read_prompt_template(f"{init_dir}/item_agent_prompt.txt")}
    all_prompts['test_ranking'] = {'user': read_prompt_template(f"{forward_dir}/user_agent_prompt.txt")}

    return all_prompts


def init(users, items, llm, init_prompts):
    user_agents = {}
    item_agents = {}
    for user in users:
        agent = AgentUser(init_prompts['user'], llm)
        user_agents[user] = agent

    for item in items:
        if len(item['item_category']) > 1:
            category = item['item_category'][1:]
        else:
            category = item['item_category']
        prompt_args = {"product_name": item['item_name'], "product_category": ', '.join(category)}
        agent = AgentItem(init_prompts['item'], llm, prompt_args)
        item_agents[item['item_id']] = agent

    return user_agents, item_agents


def forward(user_agent, negative_item_agent, positive_item_agent, forward_prompt_template):
    user_agent_memory = user_agent.short_memory
    neg_item_agent_memory = negative_item_agent.memory
    pos_item_agent_memory = positive_item_agent.memory
    args_dict = {'user_agent_memory': user_agent_memory,
                 'neg_item_agent_memory': neg_item_agent_memory,
                 'pos_item_agent_memory': pos_item_agent_memory}
    forward_user_prompt = forward_prompt_template.format(**args_dict)
    print(f"\n{forward_user_prompt=}\n")
    recommendation_text = user_agent.get_answer(forward_user_prompt)

    answer_phrase = 'Chosen CD: '
    answer = extract_text_by_phrase(recommendation_text, answer_phrase)
    explanation_phrase = 'Explanation: '
    explanation = extract_text_by_phrase(recommendation_text, explanation_phrase)
    print(f"\n{answer=}. | {explanation=}\n")
    return answer, explanation


def backward(user_agent, negative_item_agent, positive_item_agent, 
                         user_explonation, neg_item_title, pos_item_title, 
                         backward_prompts_templates):
    user_agent_memory = user_agent.short_memory
    neg_item_agent_memory = negative_item_agent.memory
    pos_item_agent_memory = positive_item_agent.memory
    args_dict = {'user_agent_memory': user_agent_memory,
                 'neg_item_agent_memory': neg_item_agent_memory,
                 'pos_item_agent_memory': pos_item_agent_memory,
                 'user_explanation': user_explonation,
                 'pos_item_title': pos_item_title,
                 'neg_item_title': neg_item_title}
    user_prompt_template = backward_prompts_templates['user']
    item_prompt_template = backward_prompts_templates['item']

    print(f"\n{user_prompt_template.format(**args_dict)=}\n")
    user_feedback = user_agent.get_answer(user_prompt_template.format(**args_dict))
    updated_memory_phrase = "My updated self-introduction:"
    updated_user_memory = extract_text_by_phrase(user_feedback, updated_memory_phrase)
    print(f"\n{updated_user_memory=}\n")
    user_agent.update_memory(updated_user_memory)

    print(f"\n{item_prompt_template.format(**args_dict)=}\n")
    items_feedback = positive_item_agent.get_answer(item_prompt_template.format(**args_dict))
    updated_pos_memory_phrase = "The updated description of the first CD is:"
    updated_neg_memory_phrase = "The updated description of the second CD is:"
    updated_pos_memory = extract_text_by_phrase(items_feedback, updated_pos_memory_phrase)
    updated_neg_memory = extract_text_by_phrase(items_feedback, updated_neg_memory_phrase)
    print(f"\n{updated_pos_memory=}\n")
    print(f"\n{updated_neg_memory=}\n")
    positive_item_agent.update_memory(updated_pos_memory)
    negative_item_agent.update_memory(updated_neg_memory)


def train_agents_memory(positive_records, negative_records_users, neutral_records_users, 
                        user_agents, item_agents, all_prompts, max_num_of_substeps, num_iterations):
    for i in tqdm(range(num_iterations)):
        cur_pos_record = positive_records[i]
        assert cur_pos_record['overall'] > 3.0
        user_id = cur_pos_record['reviewer_id']
        if negative_records_users.get(user_id):
            neg_records_user_list = negative_records_users[user_id]
            neg_record = sample(neg_records_user_list, k=1)[0]
        elif neutral_records_users.get(user_id):
            neg_records_user_list = neutral_records_users[user_id]
            neg_record = sample(neg_records_user_list, k=1)[0]    
        else:
            neg_record = find_some_record(cur_pos_record, positive_records)

        cur_user_agent = user_agents[cur_pos_record['reviewer_id']]
        cur_pos_item_agent = item_agents[cur_pos_record['asin']]
        cur_neg_item_agent = item_agents[neg_record['asin']]

        for j in range(max_num_of_substeps):
            answer, user_explonation = forward(cur_user_agent, cur_neg_item_agent, cur_pos_item_agent, 
                                               all_prompts['forward']['user'])
            is_correct = check_answer(cur_pos_record['title'], neg_record['title'], answer)

            if is_correct:
                print('CORRECT')
                break
            else:
                backward(cur_user_agent, cur_neg_item_agent, cur_pos_item_agent, 
                         user_explonation, neg_record['title'], cur_pos_record['title'], 
                         all_prompts['backward'])
                print('Not correct')
    

def get_9_additional_records(negative_records_users, neutral_records_users, user_id, record, positive_records, 
                             random_neg_item_sampling):
    neg_records = []
    if not random_neg_item_sampling and negative_records_users.get(user_id):
        neg_records_user_list = negative_records_users[user_id]
        max_sample_num = min(len(neg_records_user_list), 9)
        neg_records.extend(sample(neg_records_user_list, k=max_sample_num))
    
    if not random_neg_item_sampling and neutral_records_users.get(user_id) and len(neg_records) < 9:
        neg_records_user_list = neutral_records_users[user_id]
        max_sample_num = min(9 - len(neg_records), len(neg_records_user_list))
        neg_records.extend(sample(neg_records_user_list, k=max_sample_num))
    
    if len(neg_records) < 9:  # HARDCODE according to the article 
        # print(f"Warning! Need additional negative records on this user: {user_id}")
        neg_records.sort(key=lambda x: x['overall'], reverse=True)
        left_records_num = 9 - len(neg_records)
        additional_records = find_some_disconnected_record(record, positive_records, neg_records, left_records_num)
        neg_records.extend(additional_records)

    return neg_records


def test_agents(positive_test_records, negative_records_users, neutral_records_users, positive_records, 
                user_agents, item_agents, forward_ranking_prompt_template, random_neg_item_sampling):
    all_test_scores = [[], [], []]
    for record in tqdm(positive_test_records[:10]):
        user_id = record['reviewer_id']
        cur_user_agent = user_agents[record['reviewer_id']]
        neg_records = get_9_additional_records(negative_records_users, neutral_records_users, user_id, 
                                               record, positive_records, random_neg_item_sampling)
        
        cur_pos_item_agent_memory = item_agents[record['asin']].memory
        cur_neg_items_agents_memories = [item_agents[neg_item['asin']].memory for neg_item in neg_records]
        memory_str = f'1. \"{cur_pos_item_agent_memory}\"\n'
        for index in range(len(cur_neg_items_agents_memories)):
            memory_str += f'{index+2}. {cur_neg_items_agents_memories[index]}\n'
        forward_ranking_prompt = forward_ranking_prompt_template.format(user_agent_memory=cur_user_agent.short_memory, 
                                                                        mixed_positive_negative_items_memory=memory_str)
        ranking_text = cur_user_agent.get_answer(forward_ranking_prompt)
        if not ranking_text:
            continue
        
        uniting_record_list = [record] + neg_records
        print(f"\n{[record['title'] for record in uniting_record_list]=}")
        assert len(uniting_record_list) == 10
        print(f"\n{ranking_text=}")
        predicted_list = prepare_eval_list(uniting_record_list, ranking_text)
        print(f"\n{predicted_list=}")
        ground_truth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # HARDCODE
        scores = calculate_scores([ground_truth_list], [predicted_list])
        for index in range(len(scores)):
            all_test_scores[index].append(scores[index])

    all_mean_scores = (sum(all_test_scores[0])/len(all_test_scores[0]), sum(all_test_scores[1])/len(all_test_scores[1]), 
                       sum(all_test_scores[2])/len(all_test_scores[2]))
    print(f"\nFinished test phase\nScores: {all_mean_scores}\n")
            

def prepare_eval_list(uniting_record_list, ranking_text):
    # Check integrity
    titles = [record['title'] for record in uniting_record_list]
    assert len(titles) == 10

    eval_list = [0 for i in range(10)]
    for index in range(10):
        check_str = f"{index+1}."
        begin_index = ranking_text.find(check_str)
        if begin_index > -1:
            cutted_text = ranking_text[begin_index:]
            end_index = cutted_text.find(',')
            if end_index < 0:
                end_index = cutted_text.find('\"')
            if end_index < 0:
                end_index = cutted_text.find('\n')
            extracted_title = cutted_text[:end_index].replace('\"','').strip()
            print(f"{extracted_title=}")
            for record_ind, record in enumerate(uniting_record_list):
                if record['title'] in extracted_title:
                    eval_list[index] = record_ind + 1
                    break
        
    return eval_list


def calculate_scores(ground_truth_list, predicted_list):
    return (ndcg_score(ground_truth_list, predicted_list, k=1), ndcg_score(ground_truth_list, predicted_list, k=5),
            ndcg_score(ground_truth_list, predicted_list, k=10))


def write_file_with_memories(user_agents, path_write_to):
    user_agents_only = list(user_agents.values())
    user_profiles = []
    for user_agent in user_agents_only:
        user_profiles.append(user_agent.short_memory)

    with open(f"{path_write_to}/profiles.txt", 'w') as file_to_write:
        for user_profile in user_profiles:
            file_to_write.write(user_profile)
            file_to_write.write(f'\n{"-"*30}\n')
    print(f"Successfully writen new json file")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--path_to_data', type=str, required=True)
    parser.add_argument('--max_num_of_steps', type=int, default=40)
    parser.add_argument('--max_num_of_substeps', type=int, default=2)
    parser.add_argument('--popular_negative', action='store_true')
    parser.add_argument('--random_neg_item_sampling', action='store_true', help='if true, \
                        during test phase negative examples will be sampled randomly just to \
                        be not connected with user, rather than be examples with negative overall for this user')
    parser.add_argument('--use_gpt35', action='store_true')
    parser.add_argument('--use_mistral', action='store_true')
    parser.add_argument('--output_path', type=str, required=True)

    arguments = parser.parse_args()
    
    data = read_amazon_data(arguments.path_to_data)
    all_prompts = read_prompts(arguments.prompt_path)  # './prompts_agent_based_colab_filtration'

    users, items, records = preprocess_data(data)
    print(f"\n{len(users)=}\n{len(items)=}\n{len(records)=}")
    positive_records, negative_records_users, neutral_records_users, positive_test_records = prepare_train_test_set(records, 
                                                                                        arguments.popular_negative)

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
            raise ValueError("if you want to use openai Mistral model set MISTRAL_KEY env variable")    
    elif arguments.model_path:
        llm = LLM(arguments.model_path)
    else:
        raise ValueError("if you don't want to use openai GPT model, set --model_path")

    user_agents, item_agents = init(users, items, llm, all_prompts['init'])
    print(f"Completed initialization")
    
    num_iterations = min(len(positive_records), arguments.max_num_of_steps)

    test_agents(positive_test_records, negative_records_users, neutral_records_users, positive_records, 
                user_agents, item_agents, all_prompts['forward']['user_ranking'], arguments.random_neg_item_sampling)

    train_agents_memory(positive_records, negative_records_users, neutral_records_users, 
                        user_agents, item_agents, all_prompts, arguments.max_num_of_substeps, num_iterations)
    
    path_write_to = arguments.output_path
    write_file_with_memories(user_agents, path_write_to)
    
    test_agents(positive_test_records, negative_records_users, neutral_records_users, positive_records, 
                user_agents, item_agents, all_prompts['forward']['user_ranking'], arguments.random_neg_item_sampling)

    print('end!')
    

if __name__ == '__main__':
    main()
