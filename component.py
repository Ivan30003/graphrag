import os
from pathlib import Path


class Component:
    def __init__(self, component_name, log, working_dir) -> None:
        self.component_name = component_name
        self.is_log = log
        self.working_dir = working_dir

    def write_statistics(self, data):
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
        
        save_path = os.path.join(Path(self.working_dir), Path(f"log_{self.component_name}.txt"))
        with open(save_path, mode="w") as log_file:
            log_file.write(data_str)
