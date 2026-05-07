import json

import argparse
from utils import read_amazon_data, preprocess_data, prepare_train_test_set, write_json_file, read_json_file
import os


# def read_json_file(path):
#     with open(path, 'r') as file:
#         data = json.load(file)
#     return data


def read_write_all_profiles(path_to_profiles, output_path, train_positive_records):
    all_profiles_files = os.listdir(path_to_profiles)
    item_profiles_files = []
    user_profiles_files = []
    for file_name in all_profiles_files:
        if "data" in file_name:
            continue
        if "item" in file_name:
            item_profiles_files.append(file_name)
        elif "user" in file_name:
            user_profiles_files.append(file_name)

    item_profiles_files = sorted(item_profiles_files, key=lambda x: int(x.split("_")[-2]))
    user_profiles_files = sorted(user_profiles_files, key=lambda x: int(x.split("_")[-2]))
    # assert len(train_positive_records) == len(user_profiles_files), f"{len(train_positive_records)=} | {len(user_profiles_files)=}"
    # assert len(train_positive_records) == len(item_profiles_files), f"{len(train_positive_records)=} | {len(item_profiles_files)=}"

    # for file_name in user_profiles_files[:150]:
    #     print(f"{file_name=}")
    # for file_name in item_profiles_files[:150]:
    #     print(f"{file_name=}")
    
    for index, positive_record in enumerate(train_positive_records):
        unix_time = positive_record['unix_time']
        print(f"{index}. {unix_time=}")
        cur_user_profiles_file = user_profiles_files[index]
        cur_item_profiles_file = item_profiles_files[index]
        user_profiles_data = read_json_file(f"{path_to_profiles}{cur_user_profiles_file}")
        item_profiles_data = read_json_file(f"{path_to_profiles}{cur_item_profiles_file}")
        user_profiles_data['timestamp'] = unix_time
        item_profiles_data['timestamp'] = unix_time

        write_json_file(user_profiles_data, f"user_profiles_{index}_", output_path)
        write_json_file(item_profiles_data, f"item_profiles_{index}_", output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_data', type=str, required=True)
    parser.add_argument('--path_to_profiles', type=str)
    # parser.add_argument('--path_to_init_item_profiles', type=str)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--dataset_type', choices=['amazon_cds', 'movie_lens'], type=str, required=True)

    arguments = parser.parse_args()

    # Prepare data
    data = read_amazon_data(arguments.path_to_data)
    users, items, records = preprocess_data(data, arguments.dataset_type)
    positive_records, negative_records_users, neutral_records_users, positive_test_records = prepare_train_test_set(records)

    read_write_all_profiles(arguments.path_to_profiles, arguments.output_path, positive_records)
    print('Data prepared')
    
    
    print("END!")

if __name__ == '__main__':
    main()
