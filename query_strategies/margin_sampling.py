import numpy as np
from .strategy import Strategy
import torch 

class MarginSampling(Strategy):
    def __init__(self, dataset, net):
        super(MarginSampling, self).__init__(dataset, net)

    def query(self, n):

        unlabeled_idxs, input_ids, attention_masks, labels = self.dataset.get_unlabeled_data()
        probs = self.predict_prop_bert(input_ids, attention_masks, labels)
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
    
    def enhanced_query(self, n):

        unlabeled_idxs, input_ids, attention_masks, labels = self.dataset.get_unlabeled_data()
        probs = self.predict_prop_bert(input_ids, attention_masks, labels)
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]

        uncertainties_sorted = uncertainties.sort()

        # intialize candidates_indices with the Least Confidence sample
        selected_idxs = []
        selected_idxs.append(uncertainties_sorted[1][:1])

        # get the similarity score of the candidates_indices and other datapoints
        selected_input_ids = input_ids[selected_idxs]
        similarity_scores_df = self.dataset.get_similarity_scores(selected_input_ids)

        
        candidate_idxs = uncertainties_sorted[1].tolist()
        probabilities = uncertainties_sorted[0].tolist()

        temp = []
        temp.append(selected_idxs[0].item())
        selected_idxs = temp

        for _ in range(1,n):
            selected_idxs = self.calculate_score(input_ids, probabilities, candidate_idxs, selected_idxs, similarity_scores_df)
            
        return unlabeled_idxs[selected_idxs]
    
    def calculate_score(self, input_ids, probabilities, candidate_idxs, selected_idxs, similarity_scores_df):

        similarity_scores = []

        for candidate_idx in candidate_idxs:
            
            if candidate_idx in selected_idxs:
                similarity_scores.append(1)
                continue

            candidate_input_id = []
            candidate_input_id.append(input_ids[candidate_idx].tolist())
            selected_input_id = input_ids[selected_idxs].tolist()

            selected_row = similarity_scores_df[similarity_scores_df['sentence_1_embeddings'].isin(candidate_input_id)]
            selected_row = selected_row[selected_row['sentence_2_embeddings'].isin(selected_input_id)]

            if selected_row.empty:
                selected_row = similarity_scores_df[similarity_scores_df['sentence_1_embeddings'].isin(selected_input_id)]
                selected_row = selected_row[selected_row['sentence_2_embeddings'].isin(candidate_input_id)]

            if len(selected_row.index) > 1:
                similarity_scores.append(selected_row['similarity_score'].max())
            else:
                similarity_scores.append(selected_row.iloc[0]["similarity_score"])

        avg = np.array([probabilities, similarity_scores])
        avg = np.average(avg, axis=0)       
        avg = torch.from_numpy(avg).sort()

        selected_idxs.append(candidate_idxs[avg[1][0]])

        return selected_idxs
