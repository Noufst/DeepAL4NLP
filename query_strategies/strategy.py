import numpy as np
import torch
import nlpaug.augmenter.word as naw #https://nlpaug.readthedocs.io/en/latest/augmenter/word/word_embs.html
from sentence_transformers import SentenceTransformer, util


class Strategy:

    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

        # Augmentation models
        self.aug_substitute = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute", aug_p=0.2, top_k = 50)
        self.aug_insert = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert", aug_p=0.2, top_k = 50)
        self.similarity_model = SentenceTransformer('all-mpnet-base-v2')

    def query(self, n):
        pass

    def enhanced_query(self, n):
        pass
    
    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def augement(self, pos_idxs, enhanced_augmentation):

        # call augmentation algorithm
        augmented_samples = self.augment_based_on_similarity_score(self.dataset.X_train[pos_idxs], self.dataset.Y_train[pos_idxs], enhanced_augmentation)
        augmented_sentences = [sentence[0] for sentence in augmented_samples]
        
        # assign labels to the augemented sentences
        augmented_sentences_labels = [sentence[1] for sentence in augmented_samples]
        augmented_sentences_labels = np.array(augmented_sentences_labels)
     
        # tokenize augmented data 
        augmented_input_ids_train, augmented_attention_masks_train, augmented_labels_train = self.dataset.tokenize(augmented_sentences, augmented_sentences_labels)

        # add the augemnted sentences to the training data
        self.dataset.input_ids_train = torch.cat((self.dataset.input_ids_train, augmented_input_ids_train), 0)
        self.dataset.attention_masks_train = torch.cat((self.dataset.attention_masks_train, augmented_attention_masks_train), 0)
        self.dataset.labels_train = torch.cat((self.dataset.labels_train, augmented_labels_train), 0)

        # update labeled_idxs
        new_labeled_idxs = np.full(len(augmented_sentences_labels), True)
        labeled_idxs = np.concatenate((self.dataset.labeled_idxs, new_labeled_idxs))
        
        # update number of samples in the pool
        self.dataset.update_n_pool()
        self.dataset.labeled_idxs = np.zeros(self.dataset.n_pool, dtype=bool)
        self.dataset.labeled_idxs[labeled_idxs] = True
    
    def augment_based_on_similarity_score(self, sentences, labels, enhanced_augmentation):

        augmented_sentences = []

        for idx, sentence in enumerate(sentences):

            # split the text into sentence (review) and category
            category = sentence.split()[-1]
            sentence = sentence[:sentence.rstrip().rfind(" ")]

            label = labels[idx]

            # start augmentation algorithm
            temp_augmented_sentences = []
            if not enhanced_augmentation:
                temp_augmented_sentences = self.generate_augmented_sentences(sentence, category, label)
            else:
                if not augmented_sentences: # if list is empty
                    temp_augmented_sentences = self.generate_augmented_sentences(sentence, category, label)
                else:
                    similarity_scores = util.dot_score(self.similarity_model.encode(sentence), self.similarity_model.encode([sentence[0][:sentence[0].rstrip().rfind(" ")] for sentence in augmented_sentences]))
                    threshold = torch.tensor([0.5])
                    is_similar = torch.gt(similarity_scores, threshold)
                    if True in is_similar:
                        continue
                    else:
                        temp_augmented_sentences = self.generate_augmented_sentences(sentence, category, label)

            for i in temp_augmented_sentences:
                augmented_sentences.append(i)
            
        return augmented_sentences

    def generate_augmented_sentences(self, sentence, category, label):
        augmented_sentences = []
        augmented_sentences.append((sentence + " " + category, label))

        # augment new 5 sentences from the sentence
        for i in range(0,5):
            augmented_sentence = self.aug_substitute.augment(sentence)[0]
            augmented_sentence = self.aug_insert.augment(augmented_sentence)[0]
            augmented_sentences.append((augmented_sentence + " " + category, label))
        
        return augmented_sentences

    def train_bert(self):

        # load data
        labeled_idxs, input_ids, attention_masks, labels = self.dataset.get_labeled_data()

        all_training_stats = []

        # hyperparameters
        LEARNING_RATES = [2e-5, 3e-5, 5e-5] 
        BATCH_SIZES = [16, 32]

        # hyperparameters tuning
        for BATCH_SIZE in BATCH_SIZES:
            for LEARNING_RATE in LEARNING_RATES:
                training_stats = self.net.train_bert(input_ids, attention_masks, labels, LEARNING_RATE, BATCH_SIZE, 4) #1
                all_training_stats.append(training_stats)
        all_training_stats = [item for sublist in all_training_stats for item in sublist]

        # retrain with best hyperparameters based on validation loss
        val_losses = []
        for dict in all_training_stats:
            val_losses.append(dict["Valid. Loss"])
        best_hp = next(item for item in all_training_stats if item["Valid. Loss"] == min(val_losses))
        print()
        print("best hyperparameters:")
        print(best_hp['epoch'], best_hp['learning_rate'], best_hp['batch_size'])
        print()

        training_stats = self.net.train_bert(input_ids, attention_masks, labels, best_hp['learning_rate'], best_hp['batch_size'], best_hp['epoch'])

        return [best_hp['epoch'], best_hp['learning_rate'], best_hp['batch_size']], len(labels)

    def predict_bert(self, input_ids, attention_masks, labels):
        preds = self.net.predict_bert(input_ids, attention_masks, labels)
        return preds
    
    def predict_prop_bert(self, input_ids, attention_masks, labels):
        preds = self.net.predict_prop_bert(input_ids, attention_masks, labels)
        return preds

