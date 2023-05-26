from handlers import SemEval_Handler
from data import get_SemEval_Restaurants, get_SemEval_Laptops
from query_strategies import LeastConfidence, MarginSampling
import csv


def get_handler(name):
    if name == 'SemEval_Restaurants' or name == 'SemEval_Laptops':
        return SemEval_Handler
    else:
        raise NotImplementedError

def get_dataset(name):
    if name == 'SemEval_Restaurants':
        return get_SemEval_Restaurants(get_handler(name))
    elif name == 'SemEval_Laptops':
        return get_SemEval_Laptops(get_handler(name))
    else:
        raise NotImplementedError

def get_strategy(name):
    if name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    else:
        raise NotImplementedError

def save_results(results, dataset_name, strategy_name, with_augmentation):

    header = ['round', 'accuracy', 'precision', 'recall', 'f1', 'best_hp', 'training_size']

    # open the file in the write mode
    if with_augmentation:
        file_name = 'results/'+dataset_name+'_'+strategy_name+'_with_augmentation.csv'
    else:
        file_name = 'results/'+dataset_name+'_'+strategy_name+'.csv'
    with open(file_name, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write rows to the csv file
        writer.writerow(header)
        writer.writerows(results)
