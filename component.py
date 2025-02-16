class Component:
    def __init__(self, component_name, is_log, working_dir) -> None:
        self.component_name = component_name
        self.is_log = is_log
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
        
        with open(f"log_{self.component_name}.txt", mode="w") as log_file:
            log_file.write(data_str)
