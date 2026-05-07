import argparse
import json
from random import shuffle
from tqdm import tqdm
import sys
from pathlib import Path
from collections import defaultdict

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from files_utils import read_yaml_file


SYMBOL_COUNT_LIMIT = 3800


def resplit_text_further(passage: str):
    texts = []
    cur_text = ''
    for ind, sub_text in enumerate(passage.split('\n')):
        if len(cur_text) + len(sub_text) < SYMBOL_COUNT_LIMIT:
            cur_text += f"\n{sub_text}"
        else:
            texts.append(cur_text.strip())
            cur_text = sub_text
        if ind == len(passage.split('\n'))-1:
            texts.append(cur_text.strip())
    
    assert len(texts) > 1, f"{len(passage)} | {len(texts[0])} \n{passage}"
    for text in texts:
        assert len(text) <= SYMBOL_COUNT_LIMIT, f"TEXT{len(text)}:\n{text}"

    return texts


def check_statistics(sampled_data):
    unique_items = set([sample['asin'] for sample in sampled_data])
    unique_users = set([sample['reviewer_id'] for sample in sampled_data])
    user_records_dict = {}
    for sample in sampled_data:
        if sample['reviewer_id'] in user_records_dict:
            user_records_dict[sample['reviewer_id']] += 1
        else:
            user_records_dict[sample['reviewer_id']] = 1

    values = list(user_records_dict.values())
    values.sort(reverse=True)

    print(f"\n max items per user = {values[:5]}\n min_items_per_user = {values[-5:]}\n \
          {len(unique_items)=}\n{len(unique_users)=}\n")


def check_unique_items(raw_user_data, raw_meta_data):
    items_from_user = set([record['asin'] for record in raw_user_data])
    items_from_meta = set([record['asin'] for record in raw_meta_data])
    print("Checking items status...")
    if len(items_from_user) <= len(items_from_meta):
        print("Status: Ok, items num from meta-data is larger than items num from user data")
    else:
        print("Status: Warning! Items num from user data is larger than Items num from meta-data")

    absent = 0
    for item in items_from_user:
        if item not in items_from_meta:
            absent += 1

    print(f"{absent} num of items is absent from user's data ({absent}/{len(items_from_user)})")
    # unique_items = list(set(items))
    print(f"{len(items_from_meta)=}")


def filter_out(data):
    filtered_united_data = []
    for sample in tqdm(data):
        if sample.get('title'):
            filtered_united_data.append(sample)
    return filtered_united_data


# def find_recommended_num_records(samples_list, min_num_pos_records, recommended_num):
#     neg_records = []
#     pos_records = []
#     neutral_records = []
#     for sample in samples_list:
#         if sample['overall'] > 3 and len(pos_records) < min_num_pos_records:
#             pos_records.append(sample)
#         elif sample['overall'] < 3 and sample['popularity'] > POPULARITY_THRESHOLD:
#             neg_records.append(sample)
#         else:
#             neutral_records.append(sample)

#     # print(f"{len(neg_records)=} {len(pos_records)=} {len(neutral_records)=}")
#     num_neutral_records_to_add = min_num_pos_records - len(pos_records)
#     pos_records.extend(neutral_records[:num_neutral_records_to_add])
#     num_neg_records_to_add = recommended_num - len(pos_records)
#     pos_records.extend(neg_records[:num_neg_records_to_add])
#     return pos_records


def uniting(raw_user_data, raw_meta_data=None) -> list:
    """
    Getting list of user-item's records with meta-data included
    """
    data_item_id_dict = defaultdict(list)
    print("uniting user data and metadata")
    for record in tqdm(raw_user_data):
        key = record['asin']
        data_item_id_dict[key].append({"reviewer_name": record.get("reviewerName"), 
                                     "review_text": record.get("reviewText"), 
                                    "summary": record.get("summary"), "unix_time": record["unixReviewTime"], 
                                    "reviewer_id": record["reviewerID"], "overall": round(record["overall"]),
                                    "asin": key})

    check_unique_items(raw_user_data, raw_meta_data)

    for record in tqdm(raw_meta_data):
        key = record['asin']
        if data_item_id_dict.get(key):
            for user_record in data_item_id_dict[key]:
                user_record['category'] = record['category']
                user_record['title'] = record['title']
                user_record['price'] = record['price']
                user_record['brand'] = record['brand']
                if record.get('also_view'):
                    view_pop = len(record['also_view'])
                if record.get('also_buy'):
                    buy_pop = len(record['also_buy'])
                user_record['popularity'] = round(0.4 * view_pop + buy_pop)
    
    data_item_id_dict = list(data_item_id_dict.values())
    data_item_id_dict = [item for sublist in data_item_id_dict for item in sublist]
    cleared_united_data = data_item_id_dict

    return cleared_united_data


def read_data(user_data_path, meta_data_path):
    with open(user_data_path, 'r') as fin:
        raw_user_data = [json.loads(data_line) for data_line in fin]
    if meta_data_path:
        with open(meta_data_path, 'r') as fin:
            raw_meta_data = [json.loads(data_line) for data_line in fin]
    
    return raw_user_data, raw_meta_data


def add_pos_neg_statistic(filtered_united_data: list) -> dict:
    user_statistics = {}
    for record in tqdm(filtered_united_data):
        reviewer_id = record['reviewer_id']
        if record['overall'] > 3:
            stats_ind = 2
        elif record['overall'] == 3:
            stats_ind = 1
        else:
            stats_ind = 0
        if not user_statistics.get(reviewer_id):
            user_statistics[reviewer_id] = [0, 0, 0] # negative, neutral, positive
        user_statistics[reviewer_id][stats_ind] += 1  # record['overall']

    total_unique_users = len(set([record['reviewer_id'] for record in filtered_united_data]))
    assert len(user_statistics) == total_unique_users, f"{len(user_statistics)=} {total_unique_users=}"

    return user_statistics


def sample_required_records(filtered_united_data: list, config: dict) -> list:
    print(f"SAMPLING...")
    num_history_records = config['history_records']['num_history_records']
    only_pos = config['history_records']['only_pos']
    num_test_positive_records = config['test_records']['num_test_positive_records']
    num_test_non_positive_records = config['test_records']['num_test_non_positive_records']
    num_users = config['num_users']
    # print(f"{num_users=} {num_test_positive_records=} {num_test_non_positive_records=} {num_history_records=}")
    
    data_user_id_dict = defaultdict(list)
    for record in filtered_united_data:
        reviewer_id = record['reviewer_id']
        data_user_id_dict[reviewer_id].append(record)

    sampled_data_user_id_dict = {}
    for user_id in data_user_id_dict:
        if len(sampled_data_user_id_dict) == num_users:
            break
        user_records = sorted(data_user_id_dict[user_id], key=lambda x: int(x['unix_time']), reverse=True)
        test_pos_set = []
        test_non_pos_set = []
        train_set = []
        for record in user_records:
            # if we haven't found sufficient amount of test items
            if len(test_pos_set) < num_test_positive_records or len(test_non_pos_set) < num_test_non_positive_records:
                if record['overall'] > 3 and len(test_pos_set) < num_test_positive_records:
                    test_pos_set.append(record)
                elif record['overall'] <= 3 and len(test_non_pos_set) < num_test_non_positive_records:
                    test_non_pos_set.append(record)
            # Found records for train set
            elif len(train_set) < num_history_records:
                if not only_pos or record['overall'] > 3:
                    train_set.append(record)
            else:
                break

        # Check if total amount is sufficient
        if (len(train_set) >= num_history_records and 
        len(test_pos_set) == num_test_positive_records and 
        len(test_non_pos_set) == num_test_non_positive_records):
            sampled_data_user_id_dict[user_id] = {"test": test_pos_set + test_non_pos_set, "train": train_set}

    if len(sampled_data_user_id_dict) < num_users:
        print(f"WARNING! Not enough users. Only {len(sampled_data_user_id_dict)} were found")
    return sampled_data_user_id_dict


def sample_amazon_cds(config):
    raw_user_data, raw_meta_data = read_data(config['path_to_user_data'], config['path_to_metadata'])
    print(f"Dataset original length = {len(raw_user_data)}")

    united_data = uniting(raw_user_data, raw_meta_data)
    print(f"United dataset length = {len(united_data)}")

    filtered_united_data = filter_out(united_data)
    print(f"United dataset length after filtering = {len(filtered_united_data)}")

    # user_stats_dict = add_pos_neg_statistic(filtered_united_data)
    
    sampled_users_records = sample_required_records(filtered_united_data, config)

    # check_statistics(sampled_data)
    print("SUCCESS!")
    
    return sampled_users_records


def prepare_pipeline_input(sampled_data):
    corpus = []
    test_items = []
    index = 0
    for user_id in sampled_data:
        # TRAIN post-process
        train_data = sampled_data[user_id]['train']
        for record in train_data:
            cur_passage = record['review_text']
            cur_id = f"{record['reviewer_id']}_{record['asin']}"
            if len(cur_passage) > SYMBOL_COUNT_LIMIT:
                splitted_text = resplit_text_further(cur_passage)
                for ind, text in enumerate(splitted_text):
                    corpus.append({"passage": text, "id": f"{cur_id}_part{ind+1}", "index": index})
                    index += 1
            else:
                corpus.append({"passage": cur_passage, "id": cur_id, "index": index})
                index += 1
        # TEST post-process
        full_test_records = sampled_data[user_id]['test']
        test_data = [{'overall': record['overall'], 
                      'title': record['title'], 
                      'category': record['category'],
                      'brand': record['brand']} for record in full_test_records]
        cur_items_test_list = sorted(test_data, key=lambda x: int(x['overall']), reverse=True)
        shuffle(cur_items_test_list)
        cur_id = record['reviewer_id']
        test_items.append({"id": cur_id, "question": cur_items_test_list})
    
    return corpus, test_items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data_config', type=str, 
                        default=Path("./dataset_converters/config_cds_vinyl_sample.yaml"))
    parser.add_argument('--output_path', type=Path, required=True)
    args = parser.parse_args()
    data_config = args.path_to_data_config
    output_path = args.output_path
    config = read_yaml_file(data_config)
    
    sampled_data = sample_amazon_cds(config)

    corpus, test_items = prepare_pipeline_input(sampled_data)
    
    output_file_path = output_path / Path(f"cds_vinyl_{config['num_users']}_corpus.json")
    with open(output_file_path, 'w') as output_file:
        json.dump(corpus, output_file, indent=4)
    output_file_path = output_path / Path(f"cds_vinyl_{config['num_users']}_test_items.json")
    with open(output_file_path, 'w') as output_file:
        json.dump(test_items, output_file, indent=4)
    print(f"Successfully wrote json file")


if __name__ == '__main__':
    main()
