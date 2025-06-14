
from processors_utils.processing_utils import DatasetProcessor
from extractor_utils.triples_entities_extraction import Extractor
from processors_utils.processing_utils import BaseProcessor
from processors_utils.processing_utils import LinkMerger
from processors_utils.rule_based_hierarchy import HierarchyModule


COMPONENT_CLASSES_DICT = {"dataset_processor": DatasetProcessor,
                          "extractor": Extractor,
                          "base_processor": BaseProcessor,
                          "link_merger": LinkMerger,
                          "hierarchy_module": HierarchyModule}


class Pipeline:
    def __init__(self, config) -> None:
        self.config = config
        self.working_dir = config['working_dir']
        self.params_dict = config['pipeline']['parts']
        self.components = []
        for part_name in self.params_dict:
            component_params = self.params_dict[part_name]
            component_params['component_name'] = part_name
            component_params['working_dir'] = self.working_dir
            component = COMPONENT_CLASSES_DICT[part_name](**component_params)
            self.components.append(component)

    def launch(self):
        print(f"LAUNCH PIPELINE")
        for component in self.components:
            print(f"COMPONENT IN WORK: {component.component_name}")
            component()
