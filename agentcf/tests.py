def check_train_test_set_integrity(user_ids, train_set, test_set, neutral_records_users, negative_records_users):
    # Supposing we have 1100 records and 100 users in total. Not less than 5 positive record per user
    assert len(train_set) > 700, f'Not enough length of train set {len(train_set)}'
    assert len(test_set) == 100
    assert len(test_set) == len(user_ids)

    test_user_ids = sorted(list(set([record['reviewer_id'] for record in test_set])))
    train_user_ids = sorted(list(set([record['reviewer_id'] for record in train_set])))
    user_ids = sorted(user_ids)

    assert test_user_ids == train_user_ids, f'{test_user_ids}\n{train_user_ids}'
    assert len(test_user_ids) == 100
    assert user_ids == train_user_ids, f'{user_ids}\n{train_user_ids}'

    for record in test_set:
        assert record['overall'] > 3.0, f"Not positive record: {record}"

    collected_records_users = {}
    for record in train_set:
        if record['overall'] > 3.0:
            if record['reviewer_id'] in collected_records_users:
                collected_records_users[record['reviewer_id']].append(record)
            else:
                collected_records_users[record['reviewer_id']] = [record]

    assert len(collected_records_users) == 100
    
    collected_user_ids = []
    for user_id in collected_records_users:
        assert len(collected_records_users[user_id]) >= 8, \
        f'Not enough pos records for {user_id}: {len(collected_records_users[user_id])}'
        collected_user_ids.append(user_id)

    collected_user_ids = sorted(collected_user_ids)
    assert collected_user_ids == test_user_ids, f'{collected_user_ids}\n{test_user_ids}'
    
    print("DATA INTEGRITY IS FULL")