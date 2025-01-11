import json
import csv


def write_statistics(data, step_name):
    data_str = ""
    if type(data) == dict:
        for key in data:
            data_str = f"{data_str}\n{'*'*50}\n{key}\n{'-'*20}"
            if type(data[key]) == list:
                for sample in data[key]:
                    data_str = f"{data_str}\n{sample}"
            else:
                data_str = f"{data_str}\n{str(data[key])}"

    elif type(data) == list:
        for sample in data:
            data_str = f"{data_str}\n{sample}"
    else:
        data_str = f"{data_str}\n{str(data)}"
    
    with open(f"log_{step_name}.txt", mode="w") as log_file:
        log_file.write(data_str)


def read_dataset(path):
    import os
    if os.path.isdir(path):
        raise ValueError("Directory")
    else:  # file
        file_extension = path.split(".")[-1]
        
        with open(path) as opened_file:
            if file_extension == 'txt':
                data = [''.join(opened_file.readlines()).strip('\n')]
            elif file_extension == 'json':
                data = [json.load(opened_file, 'r')]
            else:
                raise ValueError(f"Unknown extension {file_extension}")
    write_statistics(f"amount of docs: {len(data)}", "read_dataset")
    
    return data


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data


def write_json_file(json_data, output_path):
    json.dump(json_data, open(output_path, 'w'), indent=4)
    print('OpenIE saved to', output_path)


def write_csv_file(path, data, name):
    if type(data) == dict:
        keys = data[0].keys()
        with open(path + name, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)
    elif type(data) == list:
        with open(path + name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
