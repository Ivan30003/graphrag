import json
from ast import literal_eval
import csv
from tqdm import tqdm

def read_amazon_data(data_file):
    with open(data_file, 'r') as file:
        raw_data = [json.loads(data_line) for data_line in file]
    
    return raw_data


def read_prompt_template(path_to_file):
    with open(path_to_file, "r") as file:
        lines = '\n'.join(file.readlines())

    return lines


def write_json_file(data, entity_info: str, path):
    with open(f"{path}/{entity_info}.json", 'w') as file:
        json.dump(data, file)


def read_json_file(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def extract_text_by_phrase(text: str, phrase: str, end_phrase='\n'):
    if text:
        answer_begin_index = text.find(phrase) + len(phrase)
        if text.find(phrase) > -1:
            cut_text = text[answer_begin_index:].lstrip().lstrip('\n')
            answer_end_index = cut_text.find(end_phrase)
            extracted_text_fragment = cut_text[:answer_end_index]
        else:
            extracted_text_fragment = ''
    else:
        extracted_text_fragment = ''

    return extracted_text_fragment


def read_prompts(prompts_files_dir):
    all_prompts = {}

    forward_dir = f"{prompts_files_dir}/forward"
    backward_dir = f"{prompts_files_dir}/backward"
    init_dir = f"{prompts_files_dir}/initialization"

    all_prompts['forward'] = {'user': read_prompt_template(f"{forward_dir}/user_agent_prompt.txt"), 
                              'user_ranking': read_prompt_template(f"{forward_dir}/user_agent_test_prompt.txt")}
    all_prompts['backward'] = {'user': read_prompt_template(f"{backward_dir}/user_agent_prompt.txt"),
                               'user_interests': read_prompt_template(f"{backward_dir}/user_agent_interest_prompt.txt"),
                                'item': read_prompt_template(f"{backward_dir}/item_agent_prompt.txt")}
    all_prompts['init'] = {'user': read_prompt_template(f"{init_dir}/user_agent_prompt.txt"), 
                           'item': read_prompt_template(f"{init_dir}/item_agent_prompt.txt")}
    all_prompts['test_ranking'] = {'user': read_prompt_template(f"{forward_dir}/user_agent_prompt.txt")}

    return all_prompts


def preprocess_data(data, dataset_type):
    items_ids = []
    items = []
    items_ids_users_pair = []
    users = []
    records = []
    if dataset_type == 'movie_lens':
        renamed_data = []
        for sample in data:
            renamed_dict = {'asin': sample['item_id'], 'reviewer_id': sample['user_id'], 'overall': sample['rating'],
                            'category': sample['genres'].split('|'), 'popularity': sample['popularity'], 
                            'unix_time': sample['timestamp'], 'title': sample['title']}
            renamed_data.append(renamed_dict)
    else:
        renamed_data = data

    for sample in tqdm(renamed_data):
        # if (sample['asin'], sample['reviewer_id']) not in items_ids_users_pair:
        if sample['asin'] not in items_ids:
            items.append({'item_id': sample['asin'], 'item_name': sample['title'], 
                        'item_category': sample['category'], 'overall': sample['overall'], 
                        'popularity': sample['popularity']})
            items_ids.append(sample['asin'])
        records.append(sample)
        items_ids_users_pair.append((sample['asin'], sample['reviewer_id']))
        if sample['reviewer_id'] not in users:
            users.append(sample['reviewer_id'])
        # else:
        #     print(f"{sample['reviewer_id']=}")
        #     pass

    return users, items, records


def prepare_train_test_set(records, polar_pair=True, items_per_step=2):
    negative_records_users = {}  # user - key
    neutral_records_users = {}  # user - key
    positive_records_users = {}  # user - key
    positive_train_records = []
    positive_test_records = []

    if polar_pair:
        for record in records:
            if record['overall'] < 3:
                if negative_records_users.get(record['reviewer_id']):
                    negative_records_users[record['reviewer_id']].append(record)
                else:
                    negative_records_users[record['reviewer_id']] = [record]
            elif record['overall'] > 3:
                if positive_records_users.get(record['reviewer_id']):
                    positive_records_users[record['reviewer_id']].append(record)
                else:
                    positive_records_users[record['reviewer_id']] = [record]
            elif 2 < record['overall'] < 4:
                if neutral_records_users.get(record['reviewer_id']):
                    neutral_records_users[record['reviewer_id']].append(record)
                else:
                    neutral_records_users[record['reviewer_id']] = [record]
    else:
        pass
    
    print(f"{sum([len(positive_records_users[key]) for key in positive_records_users])=}")
    for user in positive_records_users:
        positive_records_users[user].sort(key=lambda x: x['unix_time'])
        last_positive_item = positive_records_users[user].pop()
        positive_test_records.append(last_positive_item)

    for user in positive_records_users:
        positive_train_records.extend(positive_records_users[user])
    positive_train_records.sort(key=lambda x: x['unix_time'])

    # # DEBUG
    print(f"{len(records)=}")
    print(f"{len(positive_train_records)=}")
    print(f"{len(positive_test_records)=}")
    # for key in negative_records_users:
    #     print(f"{len(negative_records_users[key])=}")
    # for key in neutral_records_users:
    #     print(f"{len(neutral_records_users[key])=}")
    # # DEBUG

    return positive_train_records, negative_records_users, neutral_records_users, positive_test_records


def find_some_record(cur_pos_record, pos_records):  # Get record, which is not connected to current user
    cur_user_id = cur_pos_record['reviewer_id']
    for sample in pos_records:
        if sample['reviewer_id'] != cur_user_id:
            return sample


def find_some_disconnected_records(cur_pos_record, records, check_list_records, num):
    cur_user_id = cur_pos_record['reviewer_id']
    additional_records = []
    for record in records:
        if len(additional_records) == num:
            return additional_records

        if record['reviewer_id'] != cur_user_id and record not in check_list_records:
            additional_records.append(record)

    return additional_records


def get_9_additional_records(negative_records_users, neutral_records_users, user_id, cur_record, positive_records, 
                             random_neg_item_sampling):
    neg_records = []
    records_for_choosing = [pos_record for pos_record in positive_records]
    if random_neg_item_sampling:
        negative_records = []
        for key in negative_records_users:
            negative_records.extend(negative_records_users[key])
        for key in neutral_records_users:
            negative_records.extend(neutral_records_users[key])

        records_for_choosing.extend(negative_records)
        additional_records = find_some_disconnected_records(cur_record, records_for_choosing, neg_records, 9)
        neg_records.extend(additional_records)
    else:
        pass
        # negative_records = negative_records_users.get(user_id)
        # neutral_records = neutral_records_users.get(user_id)
        # if negative_records_users.get(user_id):
        #     max_sample_num = min(len(negative_records), 9)
        #     neg_records.extend(sample(negative_records, k=max_sample_num))
        
        # if neutral_records_users.get(user_id) and len(neg_records) < 9:
        #     max_sample_num = min(9 - len(neg_records), len(neutral_records))
        #     neg_records.extend(sample(neutral_records, k=max_sample_num))
        
        # if len(neg_records) < 9:  # HARDCODE according to the article 
        #     neg_records.sort(key=lambda x: x['overall'], reverse=True)
        #     left_records_num = 9 - len(neg_records)
        #     if random_neg_item_sampling:
        #         if negative_records:
        #             records_for_choosing += negative_records
        #         if neutral_records:
        #             records_for_choosing += neutral_records
        #     additional_records = find_some_disconnected_records(cur_record, records_for_choosing, neg_records, left_records_num)
        #     neg_records.extend(additional_records)

    return neg_records


def prepare_eval_list(uniting_record_list, ranking_text: str):
    # Check integrity
    titles = [record['title'] for record in uniting_record_list]
    assert len(titles) == 10

    try:
        start_index = ranking_text.find('[')
        end_index = ranking_text.find(']')
        assert start_index > -1 and end_index > -1, 'not found [ and ]'
        cutted_text = ranking_text[start_index: end_index+1]
        eval_list = literal_eval(cutted_text)
        assert type(eval_list) == list, 'not list'
        assert len(eval_list) == 10, 'length is not 10'
    except Exception as err:
        print(f"error: {err} in {ranking_text}")
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
                for record_ind, record in enumerate(uniting_record_list):
                    if record['title'] in extracted_title:
                        eval_list[index] = record_ind + 1
                        break
        
    return eval_list


def rewrite_dat_to_csv(path_to_dat_file):
    # read flash.dat to a list of lists
    datContent = [i.strip().split() for i in open(path_to_dat_file).readlines()]
    name = path_to_dat_file.split('/')[-1].split('.')[0]
    # write it as a new CSV file
    with open(f"{name}.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(datContent)
