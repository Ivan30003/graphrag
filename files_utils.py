import json
import csv
import yaml

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data

def read_json(file_path):
    with open(file_path, mode='r') as input_file:
        data = json.load(input_file)

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
