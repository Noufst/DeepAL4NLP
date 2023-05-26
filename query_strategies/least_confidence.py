import numpy as np
from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidence, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, input_ids, attention_masks, labels = self.dataset.get_unlabeled_data()
        probs = self.predict_prop_bert(input_ids, attention_masks, labels)
        uncertainties = probs.max(axis=1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
