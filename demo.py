import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy, save_results
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=20, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=20, help="number of queries per round")
parser.add_argument('--dataset_name', type=str, default="SemEval_Restaurants", choices=["SemEval_Restaurants", "SemEval_Laptops"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="LeastConfidence", choices=[ "LeastConfidence", "MarginSampling"], help="query strategy")
parser.add_argument('--active_learning', type=bool, default=False, help="enable or disable active learning")
parser.add_argument('--patient', type=int, default=5, help="number of round with no improvement after which active learning will be stopped")
parser.add_argument('--with_augmentation', type=bool, default=False, help="enable or disable augmentation")
parser.add_argument('--enhanced_augmentation', type=bool, default=False, help="enable or disable enhanced augmentation")

args = parser.parse_args()
pprint(vars(args))
print()

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset = get_dataset(args.dataset_name)   # load dataset
strategy = get_strategy(args.strategy_name) # load strategy

# start experiment
dataset.initialize_labels(args.n_init_labeled, args.active_learning)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

PATIENT = args.patient
results = []

# ******* start round 0 *********
print("Round 0")

# train
best_hp, training_size = strategy.train_bert()

# predict
input_ids_test, attention_masks_test, labels_test = dataset.get_test_data()
preds = strategy.predict_bert(input_ids_test, attention_masks_test, labels_test)

# evaluate
acc = dataset.cal_test_acc(preds)
precision = dataset.cal_test_precision(preds)
recall = dataset.cal_test_recall(preds)
f1 = dataset.cal_test_f1(preds)

print(f"Round 0 testing accuracy: {acc}")
print(f"Round 0 testing precision: {precision}")
print(f"Round 0 testing recall: {recall}")
print(f"Round 0 testing f1: {f1}")
results.append([0, acc, precision, recall, f1, best_hp, training_size])
# ******* end round 0 *********

# ******* start next rounds *********
if args.active_learning:

    rd = 1
    patinet = PATIENT
    while True:

        print(f"Round {rd}")

        # query
        query_idxs = strategy.query(args.n_query)

        # update labels
        strategy.update(query_idxs)
        
        # augment
        if args.with_augmentation:
            strategy.augement(query_idxs, args.enhanced_augmentation)
        
        # train
        best_hp, training_size = strategy.train_bert()
        
        # predict
        preds = strategy.predict_bert(input_ids_test, attention_masks_test, labels_test)

        # evaluate
        acc = dataset.cal_test_acc(preds)
        precision = dataset.cal_test_precision(preds)
        recall = dataset.cal_test_recall(preds)
        f1 = dataset.cal_test_f1(preds)
        print(f"Round {rd} testing accuracy: {acc}")
        print(f"Round {rd} testing precision: {precision}")
        print(f"Round {rd} testing recall: {recall}")
        print(f"Round {rd} testing f1: {f1}")
        results.append([rd, acc, precision, recall, f1, best_hp, training_size])

        rd =  rd + 1

        # early stopping
        if round(results[-2][1], 3) >= round(results[-1][1], 3):
            patinet = patinet - 1
            if patinet == 0:
                break
        else:
            patinet = PATIENT


print("Done")
# save results
save_results(results, args.dataset_name, args.strategy_name, args.with_augmentation)




