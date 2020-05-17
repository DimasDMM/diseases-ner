
class ModelBase:
    def __init__(self, file_manager, nlp_tool):
        self._file_manager = file_manager
        self._nlp_tool = nlp_tool

    def bin_pred_to_str(self, nlp_text, bin_pred):
        i = 0
        
        text_length = len(nlp_text)
        entities = []

        # Get list of entities separately
        entities_stack = []
        while i < text_length:
            if bin_pred[i]:
                entities_stack.append((i, nlp_text[i][2].text_with_ws))
            i = i + 1

        # Concatenate consecutive entities
        last_pos = None
        for e in entities_stack:
            if last_pos == e[0] - 1:
                last_pos = e[0]
                n_entities = len(entities) - 1
                entities[n_entities] = entities[n_entities] + e[1]
            else:
                last_pos = e[0]
                entities.append(e[1])

        entities = [e.strip() for e in entities]        
        
        return entities
