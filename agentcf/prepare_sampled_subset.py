import argparse
import json
import random
from tqdm import tqdm
import os
import pandas as pd


MIN_NUM_POSITIVE_RECORDS = 18
MIN_NUM_NEGATIVE_RECORDS = 2
POPULARITY_THRESHOLD = 100


AGE_DICT = {1: "Under 18",
            18: "18-24",
            25: "25-34",
            35: "35-44",
            45: "45-49",
            50: "50-55",
            56: "56+"}


def check_unique_items(raw_user_data, raw_meta_data):
    items_from_user = set([record['asin'] for record in raw_user_data])
    items_from_meta = set([record['asin'] for record in raw_meta_data])
    if len(items_from_user) <= len(items_from_meta):
        print("YES")
    else:
        print("NO")

    absent = 0
    for item in items_from_user:
        if item not in items_from_meta:
            absent += 1

    print(f"\n{absent=}\n{len(items_from_user)=}\n")
    # unique_items = list(set(items))
    print(f"\n{len(items_from_meta)=}\n")

    
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


def uniting(raw_user_data, raw_meta_data=None):
    united_data = {}
    print("uniting user data and metadata")
    for record in tqdm(raw_user_data):
        key = record['asin']
        if united_data.get(key):
            united_data[key].append({"reviewer_name": record.get("reviewerName"), 
                                     "review_text": record.get("reviewText"), 
                                    "summary": record.get("summary"), "unix_time": record["unixReviewTime"], 
                                    "reviewer_id": record["reviewerID"], "overall": round(record["overall"])})
        else:
            united_data[key] = [{"reviewer_name": record.get("reviewerName"), 
                                 "review_text": record.get("reviewText"), 
                                    "summary": record.get("summary"), "unix_time": record["unixReviewTime"], 
                                    "reviewer_id": record["reviewerID"], "overall": round(record["overall"])}]

    check_unique_items(raw_user_data, raw_meta_data)

    for record in tqdm(raw_meta_data):
        key = record['asin']
        if united_data.get(key):
            for user_record in united_data[key]:
                user_record['category'] = record['category']
                user_record['title'] = record['title']
                user_record['price'] = record['price']
                user_record['brand'] = record['brand']
                if record.get('also_view'):
                    view_pop = len(record['also_view'])
                if record.get('also_buy'):
                    buy_pop = len(record['also_buy'])
                user_record['popularity'] = round(0.4 * view_pop + buy_pop)
    
    NEED_LENGTH = 11
    united_data = list(united_data.items())

    deleted = 0
    cleared_united_data = []
    for key_dict in tqdm(united_data):
        key = key_dict[0]
        for user_record in key_dict[1]:
            if len(user_record) == NEED_LENGTH:
                user_record['asin'] = key
                cleared_united_data.append(user_record)
            else:
                if deleted < 3:
                    print(f"\n{user_record=}")
                deleted += 1
    print(f"\n{deleted=}\n")
    return cleared_united_data


def filter_out(data, required_items=None):
    filtered_united_data = []
    if required_items:
        for sample in tqdm(data):
            if '<' not in sample['title'] and sample['asin'] in required_items:
                filtered_united_data.append(sample)
    else:
        for sample in tqdm(data):
            if '<' not in sample['title']:
                filtered_united_data.append(sample)
    return filtered_united_data


def find_recommended_num_records(samples_list, min_num_pos_records, recommended_num):
    neg_records = []
    pos_records = []
    neutral_records = []
    for sample in samples_list:
        if sample['overall'] > 3 and len(pos_records) < min_num_pos_records:
            pos_records.append(sample)
        elif sample['overall'] < 3 and sample['popularity'] > POPULARITY_THRESHOLD:
            neg_records.append(sample)
        else:
            neutral_records.append(sample)

    # print(f"{len(neg_records)=} {len(pos_records)=} {len(neutral_records)=}")
    num_neutral_records_to_add = min_num_pos_records - len(pos_records)
    pos_records.extend(neutral_records[:num_neutral_records_to_add])
    num_neg_records_to_add = recommended_num - len(pos_records)
    pos_records.extend(neg_records[:num_neg_records_to_add])
    return pos_records


def sample_required_num_records(filtered_united_data, sampled_ids, num_records):
    selected_by_user_data_dict = {}
    count = 0
    for sample in filtered_united_data:
        key = sample['reviewer_id']
        if key in sampled_ids:
            if selected_by_user_data_dict.get(key):
                selected_by_user_data_dict[key].append(sample)
            else:
                selected_by_user_data_dict[key] = [sample]
            count += 1

    final_records_num = min(count, num_records)
    # selected_by_user_data = list(selected_by_user_data_dict.values())

    recommeded_num = final_records_num // len(sampled_ids) # (1.2 * num_records) // len(sampled_ids)
    final_sampled_data = []
    for user_id in selected_by_user_data_dict:
        samples_list = selected_by_user_data_dict[user_id]
        cur_user_pos_records = find_recommended_num_records(samples_list, MIN_NUM_POSITIVE_RECORDS, recommeded_num)
        final_sampled_data.extend(cur_user_pos_records)  # samples_list[0]
        # sampled_records.extend(samples_list[MIN_NUM_POSITIVE_RECORDS:recommeded_num])
    print(f"{final_records_num=}  {len(final_sampled_data)=}")
    random.shuffle(final_sampled_data)
    # final_sampled_data.extend(sampled_records[:final_records_num - len(final_sampled_data)])

    return final_sampled_data


def mapping_amazon_cds(user_data_path, meta_data_path, required_users_items_path, num_users, num_records, popular_negative):
    with open(user_data_path, 'r') as fin:
        raw_user_data = [json.loads(data_line) for data_line in fin]
    if meta_data_path:
        with open(meta_data_path, 'r') as fin:
            raw_meta_data = [json.loads(data_line) for data_line in fin]
    required_users_items = None
    if required_users_items_path:
        with open(required_users_items_path, 'r') as fin:
            required_users_items = [json.loads(data_line) for data_line in fin][0]
        print(f"required_users_items users: {len(required_users_items['user_id'])} \
          | items:{len(required_users_items['item_id'])}")
    print(f"Dataset original length = {len(raw_user_data)}")
    # print(f"{required_users_items.keys()=}")
    

    united_data = uniting(raw_user_data, raw_meta_data)

    if required_users_items:
        filtered_united_data = filter_out(united_data, required_users_items['item_id'])
    else:
        filtered_united_data = filter_out(united_data)

    user_statistics = {}
    for record in tqdm(filtered_united_data):
        key = record['reviewer_id']
        if record['overall'] > 2:
            overall_ind = 1
        elif record['popularity'] > POPULARITY_THRESHOLD or not popular_negative:
            overall_ind = 0
        else:
            overall_ind = None
        if not user_statistics.get(key):
            user_statistics[key] = [0, 0, 0] # negative, positive overall and popularity > 30 counts
        if record['popularity'] > POPULARITY_THRESHOLD:
            user_statistics[key][2] += 1
        if overall_ind is not None:
            user_statistics[key][overall_ind] += 1  # record['overall']

    total_unique_users = len(set([record['reviewer_id'] for record in filtered_united_data]))
    assert len(user_statistics) == total_unique_users, f"{len(user_statistics)=} {total_unique_users=}"
    sampled_ids = []
    if not required_users_items:
        for user in user_statistics:
            # print(f"{user_statistics[user]=}")
            if user_statistics[user][0] > MIN_NUM_NEGATIVE_RECORDS and user_statistics[user][1] > MIN_NUM_POSITIVE_RECORDS:
                sampled_ids.append(user)
                if len(sampled_ids) == num_users:
                    break
        assert len(sampled_ids) == num_users, f"Not enough users {len(sampled_ids)}. Need {num_users}"
        sampled_records_users = {}
        for record in filtered_united_data:
            if record['reviewer_id'] in sampled_ids:
                if record['reviewer_id'] in sampled_records_users:
                    sampled_records_users[record['reviewer_id']].append(record)
                else:
                    sampled_records_users[record['reviewer_id']] = [record]

        print(f"{len(sampled_records_users)=}")
        recommended_num = num_records // num_users
        sampled_data = []
        for user_id in sampled_records_users:
            sampled_data_one_user = find_recommended_num_records(sampled_records_users[user_id], 
                                                        MIN_NUM_POSITIVE_RECORDS, recommended_num)
            assert len(sampled_data_one_user) == recommended_num, f"{len(sampled_data_one_user)=} not {recommended_num}"
            sampled_data.extend(sampled_data_one_user)

        print(f"{len(sampled_ids)=}; {num_users=}\n")

        check_statistics(sampled_data)
        print(f"-------------------\n\n")
        for sample in sampled_data:
            print(f"{sample['reviewer_id']}___{sample['asin']}")
        return sampled_data
    else:
        required_users = required_users_items['user_id']
        required_items = required_users_items['item_id']

        sampled_ids = []
        for user in required_users:
            if user_statistics.get(user):
                if user_statistics[user][0] >= MIN_NUM_NEGATIVE_RECORDS:
                    sampled_ids.append(user)

        print(f"{len(required_users)=} | {len(sampled_ids)=}")

        sampled_data = []
        for record in tqdm(filtered_united_data):
            if record['reviewer_id'] in sampled_ids and record['asin'] in required_items:
                sampled_data.append(record)

        print(f"Sampled {len(sampled_data)} records in total")

        return sampled_data


def check_base_movies_len(ratings_df, users_df, movies_df):
    print(f"Total length = {len(ratings_df)}")
    print(f"Unique users in ratings {len(pd.unique(ratings_df['user_id']))} | in users {len(pd.unique(users_df['user_id']))}")
    print(f"Unique items in ratings {len(pd.unique(ratings_df['item_id']))} | in movies {len(pd.unique(movies_df['item_id']))}")
    print(f"\nNulls in ratings:")
    print(f"{ratings_df.isnull().sum()}")
    print(f"\nNulls in users:")
    print(f"{users_df.isnull().sum()}")
    print(f"\nNulls in movies:")
    print(f"{movies_df.isnull().sum()}")
    print(f"{'-'*40}")


def check_movie_df_for_sampling(sampled_ratings_df, num_records, num_users):
    sampled_num_records = len(sampled_ratings_df)
    unique_users_num = len(pd.unique(sampled_ratings_df['users_id']))
    assert sampled_num_records >= num_records, f"Not enough data: {sampled_num_records=}, needed {num_records}"
    assert unique_users_num >= num_users, f"Not enough data: {unique_users_num=}, needed {num_users}"
    pos_rec_num = sampled_num_records[sampled_num_records['rating'] > 3.5]


def mapping_movie_lens(data_dir_path, num_users, num_records, popular_negative):
    assert os.path.isdir(data_dir_path)
    # assert os.path.isfile(user_data_path)
    RATINGS = "ratings.dat"
    USERS = "users.dat"
    MOVIES = "movies.dat"
    POPULARITY_THRESHOLD_MOVIES = 3000
    recommended_num_records = num_records // num_users
    min_neg_items_num = recommended_num_records - MIN_NUM_POSITIVE_RECORDS

    ratings_df = pd.read_csv(f"{data_dir_path}{RATINGS}", names=['user_id', 'item_id', 'rating', 'timestamp'], 
                             sep='::', engine="python", encoding='latin-1')
    users_df = pd.read_csv(f"{data_dir_path}{USERS}", names=["user_id","gender","age","occupation","zip_code"], 
                           sep='::', engine="python", encoding='latin-1')
    movies_df = pd.read_csv(f"{data_dir_path}{MOVIES}", names=["movie_id", "title", "genres"], 
                            sep='::', engine="python", encoding='latin-1')
    movies_df.rename(columns={"movie_id": "item_id"}, inplace=True)
    # DEBUG
    print(f"{ratings_df.head()}\n")
    print(f"{users_df.head()}\n")
    print(f"{movies_df.head()}\n")
    check_base_movies_len(ratings_df, users_df, movies_df)

    ratings_users = ratings_df.merge(users_df, on="user_id")
    print(f"{ratings_users.head()}\n")
    merged_data_df = ratings_users.merge(movies_df, on="item_id")
    # DEBUG
    print(f"{merged_data_df.head()}\n")
    print(f"\nNulls in merged_data:")
    print(f"{merged_data_df.isnull().sum()}")

    merged_data_df['popularity'] = merged_data_df['rating'].groupby(merged_data_df['item_id']).transform('count') * \
    (merged_data_df['rating'].groupby(merged_data_df['item_id']).transform('mean'))**0.2
    print(f"\n{merged_data_df['popularity'].head(10)}\n")

    sampled_ratings_df = merged_data_df[(merged_data_df['rating'] < 2.5) & 
                                        (merged_data_df['popularity'] > POPULARITY_THRESHOLD_MOVIES) | 
                                        (merged_data_df['rating'] > 3.5)]
    
    # sampled_ratings_df['pos_records_num'] = sampled_ratings_df[sampled_ratings_df['rating'] > 3.5]. \
    # groupby(sampled_ratings_df['user_id']).transform('count')
    sampled_data = sampled_ratings_df.to_dict('records')
    sampled_data_user_pos_neg = {}
    for record in sampled_data:  # POS NEG
        cur_user_id = record['user_id']
        if not sampled_data_user_pos_neg.get(cur_user_id):
            sampled_data_user_pos_neg[cur_user_id] = [[],[]]
        if record['rating'] > 3.5:
            sampled_data_user_pos_neg[cur_user_id][0].append(record)
        elif record['rating'] < 2.5 and record['popularity'] > POPULARITY_THRESHOLD_MOVIES:
            sampled_data_user_pos_neg[cur_user_id][1].append(record)
        else:
            raise ValueError(f"! {record=}")

    sampled_records = []
    for user_id in sampled_data_user_pos_neg:
        pos_records_list = sampled_data_user_pos_neg[user_id][0]
        neg_records_list = sampled_data_user_pos_neg[user_id][1]
        
        if len(pos_records_list) > MIN_NUM_POSITIVE_RECORDS and len(neg_records_list) > min_neg_items_num:
            sampled_pos_rec = random.sample(pos_records_list, MIN_NUM_POSITIVE_RECORDS)
            sampled_neg_rec = random.sample(neg_records_list, min_neg_items_num)
            sampled_records.extend(sampled_neg_rec)
            sampled_records.extend(sampled_pos_rec)

        if len(sampled_records) == num_records:
            break
            
    assert len(sampled_records) == num_records, f"Not correct length: {len(sampled_records)}, needed {num_records}"

    return sampled_records


def mapping(dataset_type, user_data_path, meta_data_path, num_users, num_records, 
            popular_negative=False, required_users_items_path=None):
    if dataset_type == 'amazon_cd':
        sampled_data = mapping_amazon_cds(user_data_path, meta_data_path, required_users_items_path,
                                          num_users, num_records, popular_negative)
    elif dataset_type == "movie_lens":
        sampled_data = mapping_movie_lens(user_data_path, num_users, num_records, popular_negative)
    else:
        raise ValueError()

    return sampled_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', choices=['amazon_cd', 'movie_lens'], required=True)
    parser.add_argument('--path_to_user_data', type=str, required=True)
    parser.add_argument('--path_to_metadata', type=str)
    parser.add_argument('--path_to_required', type=str)
    parser.add_argument('--path_write_to', type=str, required=True)
    parser.add_argument('--num_users', type=int, default=5)
    parser.add_argument('--num_records', type=int, default=25)
    parser.add_argument('--popular_negative', action='store_true')
    arguments = parser.parse_args()
    
    sampled_data = mapping(arguments.dataset_type, arguments.path_to_user_data, arguments.path_to_metadata,
                           arguments.num_users, arguments.num_records, arguments.popular_negative,
                           arguments.path_to_required)
    
    with open(arguments.path_write_to, 'w') as file_to_write:
        for data in sampled_data:
            file_to_write.write(json.dumps(data))
            file_to_write.write('\n')
    print(f"Successfully writen new json file")


if __name__ == '__main__':
    main()
