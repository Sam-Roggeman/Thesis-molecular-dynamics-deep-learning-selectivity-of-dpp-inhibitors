
class LabelEncoder:
    label_mapping = {
        "nonbinder": 0,
        "dpp9selective": 1,
        "dpp8selective": 2,
        "aselective": 3,
        "apo": 4
    }
    def encode_labels(self,labels):
        labels = [self.label_mapping[label] for label in labels]
        return labels
    def decode_labels(self,encoded_labels):
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        labels = [reverse_mapping[encoded_label] for encoded_label in encoded_labels]
        return labels

    def get_classes(self):
        return list(self.label_mapping.keys())