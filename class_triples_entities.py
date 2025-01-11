class Triple:
    def __init__(self, first_entity, linkage_entity, second_entity):
        self.length = 0
        self.entities = []
        if first_entity and type(first_entity) == str:
            self.first_entity = first_entity
            self.length += 1
            self.entities.append(first_entity)
        else:
            self.first_entity = "_"
        
        if linkage_entity and type(linkage_entity) == str:
            self.linkage_entity = linkage_entity
            self.length += 1
        else:
            self.linkage_entity = "_"
        
        if second_entity and type(second_entity) == str:
            self.second_entity = second_entity
            self.length += 1
            self.entities.append(second_entity)
        else:
            self.second_entity = "_"

    def __repr__(self) -> str:
        return f"{self.first_entity} - {self.linkage_entity} - {self.second_entity}"

    def __len__(self) -> int:
        return 

    # def check_integrity(self):
    #     if self.first_entity and self.linkage_entity and self.second_entity:
    #         return True
    #     else:
    #         return False