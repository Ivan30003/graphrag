class Triple:
    def __init__(self, first_entity: str, linkage_entity: str, second_entity: str, text_idx):
        self.length = 0
        self.first_entity = first_entity
        self.linkage = linkage_entity
        self.second_entity = second_entity
        self.text_idx = text_idx
        self.content_list = [first_entity, linkage_entity, second_entity]
        self.table_represent = self.content_list + [self.text_idx]

    def __eq__(self, other):
        return self.first_entity == other.first_entity and self.linkage == other.linkage and \
        self.second_entity == other.second_entity

    def __hash__(self):
        return hash(tuple([self.first_entity, self.linkage, self.second_entity]))


class ExpandTriple:
    def __init__(self, content_list, text_idx):
        self.length = 0
        if type(content_list) == str:
            self.content_list = [content_list]
        elif type(content_list) == dict:
            self.content_list = [content_list]
        else:
            self.content_list = content_list

        self.text_idx = text_idx
        self.table_represent = self.content_list + [self.text_idx]

    def __eq__(self, other):
        return self.first_entity == other.first_entity and self.linkage == other.linkage and \
        self.second_entity == other.second_entity

    def __hash__(self):
        return hash(tuple(self.content_list))
