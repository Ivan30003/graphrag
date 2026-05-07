from agent_cf import AgentUser, AgentItem
from utils import extract_text_by_phrase, find_some_record, prepare_eval_list, get_9_additional_records, write_json_file
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from random import sample
from concurrent.futures import ThreadPoolExecutor, as_completed


class AgentCFSystem:
    def __init__(self, users, items, llm, init_prompts, dataset_type, init_users_profiles: dict, 
                 init_items_profiles: dict, path_write_to) -> None:
        user_agents = {}
        item_agents = {}
        if dataset_type == 'movie_lens':
            category_shift = 0
            self.key_word = 'movie'
        else:
            category_shift = 1
            self.key_word = 'CD'
        for user in users:
            if init_users_profiles:
                agent = AgentUser(init_users_profiles[user], llm)
            else:
                agent = AgentUser(init_prompts['user'], llm)
            user_agents[user] = agent

        for item in items:
            if init_items_profiles:
                agent = AgentItem(init_items_profiles[item['item_id']], llm)
            else:
                if len(item['item_category']) > 1:
                    category = item['item_category'][category_shift:]
                else:
                    category = item['item_category']
                prompt_args = {"product_name": item['item_name'], "product_category": ', '.join(category)}
                agent = AgentItem(init_prompts['item'].format(**prompt_args), llm)
            item_agents[item['item_id']] = agent

        self.user_agents = user_agents
        self.item_agents = item_agents
        self.path_write_to = path_write_to
        self.llm = llm

    def save(self, profiles: dict, cur_unix_time: int, cur_step, entity, path_write_to):
        profiles['step'] = cur_step
        profiles['unix_time'] = cur_unix_time
        write_json_file(profiles, f"{entity}_{cur_step}_", path_write_to)

    def check_answer(self, pos_item_title, neg_item_title, answer):
        if pos_item_title in answer and neg_item_title not in answer:
            return True
        else:
            return False

    def forward(self, user_agent, negative_item_agent, positive_item_agent, forward_prompt_template):
        user_agent_memory = user_agent.short_memory
        neg_item_agent_memory = negative_item_agent.memory
        pos_item_agent_memory = positive_item_agent.memory
        args_dict = {'user_agent_memory': user_agent_memory,
                    'neg_item_agent_memory': neg_item_agent_memory,
                    'pos_item_agent_memory': pos_item_agent_memory}
        forward_user_prompt = forward_prompt_template.format(**args_dict)
        recommendation_text = user_agent.get_answer(forward_user_prompt, temperature=0.0)

        # REPEAT 2 times
        answer_phrase = f'Chosen {self.key_word}: '
        answer = extract_text_by_phrase(recommendation_text, answer_phrase)
        explanation_phrase = 'Explanation: '
        explanation = extract_text_by_phrase(recommendation_text, explanation_phrase)
        # if answer == '' or explanation == '':
        #     recommendation_text = user_agent.get_answer(forward_user_prompt, temperature=0.2)
        #     answer_phrase = f'Chosen {self.key_word}: '
        #     answer = extract_text_by_phrase(recommendation_text, answer_phrase)
        #     explanation_phrase = 'Explanation: '
        #     explanation = extract_text_by_phrase(recommendation_text, explanation_phrase)
        
        return answer, explanation

    def backward(self, user_agent, negative_item_agent, positive_item_agent, 
                            user_explonation, neg_item_title, pos_item_title, 
                            backward_prompts_templates):
        user_agent_memory = user_agent.short_memory
        user_preferences = user_agent.preferences
        neg_item_agent_memory = negative_item_agent.memory
        pos_item_agent_memory = positive_item_agent.memory

        args_dict = {'user_agent_memory': user_agent_memory,
                    'neg_item_agent_memory': neg_item_agent_memory,
                    'pos_item_agent_memory': pos_item_agent_memory,
                    'user_explanation': user_explonation,
                    'pos_item_title': pos_item_title,
                    'neg_item_title': neg_item_title,
                    'user_preferences': user_preferences}
        user_prompt_template = backward_prompts_templates['user']
        user_interests_prompt_template = backward_prompts_templates['user_interests']
        item_prompt_template = backward_prompts_templates['item']

        user_interests_text = user_agent.get_answer(user_interests_prompt_template.format(**args_dict), temperature=0.0)
        updated_preferences_phrase = 'My updated general preferences:'
        updated_user_preferences = extract_text_by_phrase(user_interests_text, updated_preferences_phrase)
        if updated_user_preferences != '':
            user_agent.update_preferences(updated_user_preferences)

        user_feedback = user_agent.get_answer(user_prompt_template.format(**args_dict), temperature=0.0)
        updated_memory_phrase = "My updated self-introduction:"
        updated_user_memory = extract_text_by_phrase(user_feedback, updated_memory_phrase)
        if updated_user_memory != '':
            user_agent.update_memory(updated_user_memory)
        else: # SECOND CALL
            user_feedback = user_agent.get_answer(user_prompt_template.format(**args_dict), temperature=0.2)
            updated_memory_phrase = "My updated self-introduction:"
            updated_user_memory = extract_text_by_phrase(user_feedback, updated_memory_phrase)
            if updated_user_memory != '':
                user_agent.update_memory(updated_user_memory)

        items_feedback = positive_item_agent.get_answer(item_prompt_template.format(**args_dict), temperature=0.0)
        updated_pos_memory_phrase = f"The updated description of the first {self.key_word} is:"
        updated_neg_memory_phrase = f"The updated description of the second {self.key_word} is:"
        updated_pos_memory = extract_text_by_phrase(items_feedback, updated_pos_memory_phrase)
        updated_neg_memory = extract_text_by_phrase(items_feedback, updated_neg_memory_phrase)
        if updated_pos_memory != '':
            positive_item_agent.update_memory(updated_pos_memory)
        if updated_neg_memory != '':
            negative_item_agent.update_memory(updated_neg_memory)
        # if updated_pos_memory == '' or updated_neg_memory == '': # SECOND CALL
        #     items_feedback = positive_item_agent.get_answer(item_prompt_template.format(**args_dict), temperature=0.2)
        #     updated_pos_memory_phrase = f"The updated description of the first {self.key_word} is:"
        #     updated_neg_memory_phrase = f"The updated description of the second {self.key_word} is:"
        #     updated_pos_memory = extract_text_by_phrase(items_feedback, updated_pos_memory_phrase)
        #     updated_neg_memory = extract_text_by_phrase(items_feedback, updated_neg_memory_phrase)
        #     if updated_pos_memory != '':
        #         positive_item_agent.update_memory(updated_pos_memory)
        #     if updated_neg_memory != '':
        #         negative_item_agent.update_memory(updated_neg_memory)

        if updated_user_memory == '' and updated_pos_memory == '' and updated_neg_memory == '':
            return False
        else:
            return True

    def train_step(self, cur_pos_record, negative_records_users, neutral_records_users, positive_records, 
                   all_prompts, max_num_of_substeps):
        user_id = cur_pos_record['reviewer_id']
        if negative_records_users.get(user_id):
            neg_records_user_list = negative_records_users[user_id]
            neg_record = sample(neg_records_user_list, k=1)[0]
        elif neutral_records_users.get(user_id):
            neg_records_user_list = neutral_records_users[user_id]
            neg_record = sample(neg_records_user_list, k=1)[0]    
        else:
            neg_record = find_some_record(cur_pos_record, positive_records)

        cur_user_agent = self.user_agents[cur_pos_record['reviewer_id']]
        cur_pos_item_agent = self.item_agents[cur_pos_record['asin']]
        cur_neg_item_agent = self.item_agents[neg_record['asin']]

        is_correct = False
        success = []
        print(f"SUBPROCESS: NEG {neg_record['title']} POS {cur_pos_record['title']}")
        
        for _ in range(max_num_of_substeps):
            answer, user_explonation = self.forward(cur_user_agent, cur_neg_item_agent, cur_pos_item_agent, 
                                            all_prompts['forward']['user'])
            if answer == '':
                answer = 'no answer were given'
            print(f"\n!!!!!CORRECT_ANSWER: {cur_pos_record['title']}\n!!!!!EXTRACTED_ANSWER: {answer}")
            is_correct = self.check_answer(cur_pos_record['title'], neg_record['title'], answer)

            if is_correct:
                break
            else:
                substep_success = self.backward(cur_user_agent, cur_neg_item_agent, cur_pos_item_agent, 
                        user_explonation, neg_record['title'], cur_pos_record['title'], 
                        all_prompts['backward'])
                success.append(substep_success)
        
        if not any(success):
            return 1
        else:
            return 0

    def train_agents_memory(self, positive_records, negative_records_users, neutral_records_users, all_prompts, 
                            begin_num, num_iterations, max_num_of_substeps, write_each_num_steps, num_threads):
        all_steps_profiles = []
        failed_cases_num = 0
        if not begin_num:
            begin_num = 0
        
        if num_threads > 1:
            print("MULTI-THREADS")
            # Divide into batches
            positive_records_batchs = [[]]
            users_met = []
            items_met = []
            for record in positive_records:
                cur_user = record['reviewer_id']
                cur_item = record['asin']
                if cur_user in users_met or cur_item in items_met:
                    positive_records_batchs.append([record])
                    users_met = []
                    items_met = []
                else:
                    positive_records_batchs[-1].append(record)
                users_met.append(cur_user)
                items_met.append(cur_item)
            
            for batch in positive_records_batchs:
                for record in batch:
                    print(f"{record['unix_time']}")
                print("---------------")

            assert 1==2
            lengths = [len(positive_records_batch) for positive_records_batch in positive_records_batchs]
            batch_stats = {}
            for batch_length in lengths:
                if batch_stats.get(batch_length):
                    batch_stats[batch_length] += 1
                else:
                    batch_stats[batch_length] = 1
            min_batch_len = min([len(positive_records_batch) for positive_records_batch in positive_records_batchs])
            print(f"{min_batch_len=} | {len(positive_records_batchs)=} \n {batch_stats.items()=}")

            num_iterations = min(num_iterations, len(positive_records_batchs)) - begin_num
            
            # TRAIN LOOP
            for index in tqdm(range(begin_num, begin_num + num_iterations)):
                cur_batch = positive_records_batchs[index]
                print(f"CURRENT_BATCH: {index}")
                for pos_record in cur_batch:
                    print(f"POS {pos_record['title']}")
                
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(self.train_step, cur_pos_record, negative_records_users, 
                                               neutral_records_users, positive_records, all_prompts, 
                                               max_num_of_substeps) for cur_pos_record in cur_batch]

                    # Process the futures as they complete
                    for future in as_completed(futures):
                        failed_cases = future.result()
                        if type(failed_cases) == int:
                            failed_cases_num += failed_cases
                        elif type(failed_cases) == list:
                            failed_cases_num += sum(failed_cases)
                        else:
                            raise ValueError()
                    
                # Saving
                if index % write_each_num_steps == 0:
                    cur_pos_record = cur_batch[-1]
                    cur_profiles = {}
                    for user_id in self.user_agents:
                        cur_profiles[user_id] = {"memory": self.user_agents[user_id].short_memory, 
                                                 "general_preferences": self.user_agents[user_id].preferences}

                    cur_item_profiles = {}
                    for item_id in self.item_agents:
                        cur_item_profiles[item_id] = self.item_agents[item_id].memory

                    cur_unix_time = cur_pos_record['unix_time']
                    self.save(cur_profiles, cur_unix_time, index+1, 'users_profiles', self.path_write_to)
                    self.save(cur_item_profiles, cur_unix_time, index+1, 'items_profiles', self.path_write_to)
                    all_steps_profiles.append(cur_profiles)

                # print(f"{'-'*50}")

        else:
            print("WITHOUT MULTI-THREADS")
            # TRAIN LOOP
            for index in tqdm(range(begin_num, begin_num + num_iterations)):
                cur_pos_record = positive_records[index]
                print(f"CURRENT_STEP: {index}")
                failed = self.train_step(cur_pos_record, negative_records_users, 
                                               neutral_records_users, positive_records, all_prompts, 
                                               max_num_of_substeps)
                failed_cases_num += failed

                # Saving
                if index % write_each_num_steps == 0:
                    cur_profiles = {}
                    for user_id in self.user_agents:
                        cur_profiles[user_id] = {"memory": self.user_agents[user_id].short_memory, 
                                                 "general_preferences": self.user_agents[user_id].preferences}

                    cur_item_profiles = {}
                    for item_id in self.item_agents:
                        cur_item_profiles[item_id] = self.item_agents[item_id].memory

                    cur_unix_time = cur_pos_record['unix_time']
                    self.save(cur_profiles, cur_unix_time, index+1, 'users_profiles', self.path_write_to)
                    self.save(cur_item_profiles, cur_unix_time, index+1, 'items_profiles', self.path_write_to)
                    all_steps_profiles.append(cur_profiles)

        print(f"\nFailed {failed_cases_num} steps from {num_iterations}\n")

    def calculate_scores(self, ground_truth_list, predicted_list):
        return (ndcg_score(ground_truth_list, predicted_list, k=1), ndcg_score(ground_truth_list, predicted_list, k=5),
                ndcg_score(ground_truth_list, predicted_list, k=10))

    def test_agents(self, negative_records_users, neutral_records_users, positive_records, positive_test_records, 
                    forward_ranking_prompt_template, random_neg_item_sampling):
        all_test_scores = [[],[],[]]  # NDCG@1, NDCG@5, NDCG@10
        test_results = {}
        for record in tqdm(positive_test_records):
            user_id = record['reviewer_id']
            cur_user_agent = self.user_agents[record['reviewer_id']]
            neg_records = get_9_additional_records(negative_records_users, neutral_records_users, user_id, 
                                                record, positive_records, random_neg_item_sampling)
            
            cur_pos_item_agent_memory = self.item_agents[record['asin']].memory
            cur_neg_items_agents_memories = [self.item_agents[neg_item['asin']].memory for neg_item in neg_records]
            memory_str = f'1. \"{cur_pos_item_agent_memory}\"\n'
            for index in range(len(cur_neg_items_agents_memories)):
                memory_str += f'{index+2}. \"{cur_neg_items_agents_memories[index]}\"\n'
            
            forward_ranking_prompt = forward_ranking_prompt_template.format(user_agent_memory=cur_user_agent.short_memory, 
                                                                            mixed_positive_negative_items_memory=memory_str)
            ranking_text = cur_user_agent.get_answer(forward_ranking_prompt, temperature=0.0)
            if not ranking_text:
                ranking_text = cur_user_agent.get_answer(forward_ranking_prompt, temperature=0.2)
                if not ranking_text:
                    continue

            uniting_record_list = [record] + neg_records
            assert len(uniting_record_list) == 10, f'{len(uniting_record_list)=}'
            predicted_list = prepare_eval_list(uniting_record_list, ranking_text)
            ground_truth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            scores = self.calculate_scores([ground_truth_list], [predicted_list])
            test_results[user_id] = scores
            for index in range(len(scores)):
                all_test_scores[index].append(scores[index])

        all_mean_scores = (sum(all_test_scores[0])/len(all_test_scores[0]), sum(all_test_scores[1])/len(all_test_scores[1]), 
                        sum(all_test_scores[2])/len(all_test_scores[2]))
        
        test_results['total'] = all_mean_scores

        return test_results

    def prepare_eval_list(self, uniting_record_list, ranking_text):
        # Check integrity
        titles = [record['title'] for record in uniting_record_list]
        print(f"\n{titles}\n{ranking_text}\n")
        eval_list = [0 for i in range(10)]
        for index in range(10):
            check_str = f"{index+1}."
            begin_index = ranking_text.find(check_str)
            if begin_index > -1:
                cutted_text = ranking_text[begin_index:]
                end_index = cutted_text.find('\n')
                extracted_title = cutted_text[:end_index].replace('\"','').strip()
                for record_ind, record in enumerate(uniting_record_list):
                    if record['title'] in extracted_title:
                        eval_list[index] = record_ind + 1
                        break
            
        return eval_list
