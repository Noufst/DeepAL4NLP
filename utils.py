from handlers import SemEval_Handler, AWARE_Handler
from data import get_SemEval_Restaurants, get_SemEval_Laptops, get_AWARE
from query_strategies import LeastConfidence, MarginSampling
import csv
from nets import Net

def get_handler(name):
    if name == 'SemEval_Restaurants' or name == 'SemEval_Laptops':
        return SemEval_Handler
    if name == 'AWARE':
        return AWARE_Handler
    else:
        raise NotImplementedError

def get_dataset(name):
    if name == 'SemEval_Restaurants':
        return get_SemEval_Restaurants(get_handler(name))
    elif name == 'SemEval_Laptops':
        return get_SemEval_Laptops(get_handler(name))
    elif name == 'AWARE':
        return get_AWARE(get_handler(name))
    else:
        raise NotImplementedError

def get_net(device):
    return Net(device)

def get_strategy(name):
    if name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    else:
        raise NotImplementedError

def save_results(results, dataset_name, strategy_name, with_augmentation, enhanced_active_learning, file_name):

    header = ['round', 'accuracy', 'precision', 'recall', 'f1', 'cm', 'best_hp', 'training_size']

    # open the file in the write mode
    if enhanced_active_learning & with_augmentation:
        file_name = 'results/'+dataset_name+'_'+strategy_name+'_with_enhanced_active_learning_augmentation_'+file_name+'.csv'
    elif with_augmentation:
        file_name = 'results/'+dataset_name+'_'+strategy_name+'_with_augmentation_'+file_name+'.csv'
    elif enhanced_active_learning:
        file_name = 'results/'+dataset_name+'_'+strategy_name+'_with_enhanced_active_learning_'+file_name+'.csv'
    else:
        file_name = 'results/'+dataset_name+'_'+strategy_name+'_'+file_name+'.csv'
        
    with open(file_name, 'a') as f:
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(results)
