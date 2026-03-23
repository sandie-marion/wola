from os.path import join, exists 
import matplotlib.pyplot as plt 
from os import mkdir 
import itertools
import torch 

import pandas as pd
import numpy as np

from time import sleep 


def get_save_file_name(kwargs):
    
    exclude = {
        'n_classes', 'attack_parameters', 'aggregator_parameters',
        'pre_aggregator_parameters', 'criterion_parameters',
        'n_step', 'reg_param', 'clip_param', 'lr', 'experiment_folder', 'n_experiments',
        'heterogeneous_distribution', 'seed', 'training_parameters'
    }
    
    experiment_id = '_'.join(str(v) for k, v in kwargs.items() if k not in exclude)
    experiment_id += '.png'

    return experiment_id


def get_all_combinations (dictionary) : 
    combinations = []
    keys = list(dictionary)
    for values in itertools.product(*map(dictionary.get, keys)):
        combinations.append(dict(zip(keys, values)))
    return combinations 


def get_mean_time_execution (path) : 
    try : 
        data = torch.load(path) 
        tab = data["Time"]
        n = len(tab) 
        mean_time = sum(tab[1:])/(n-1)
        return mean_time 
    except :
        return None 


def get_mean_accuracy (path) : 
    try : 
        data = torch.load(path) 
        accuracy = data["Accuracy"]
        backward_acc = accuracy[::-1]
        mean_acc = (sum(backward_acc[0:5])/5)*100
        return mean_acc 
    except :
        return None 
    

def get_mean_filter_score (path, n, f) : 
    data = torch.load(path) 
    score = data["filter_score"]

    mean_sc = sum(score)/len(score)
    byz = (n-f) - mean_sc
    return byz 


def get_file_name (dataset, agg, pre_agg, alpha, attack, f, const) : 
    experiment_id = f"{attack}_" 
    experiment_id += f"{agg}_" 
    experiment_id += f"{pre_agg}_" 
    if agg == "HullGuard" : 
        experiment_id += "random_"
    experiment_id += f"{const["criterion_name"]}_" 
    experiment_id += f"{dataset}_" 
    experiment_id += f"{f}_" 
    experiment_id += f"{alpha}_" 
    experiment_id += f"{const["n_workers"]}_" 
    experiment_id += f"{const["batch_size"]}_" 
    experiment_id += f"{const["beta"]}" 
    experiment_id += '_statistics.pt'

    return experiment_id

def get_filter_file_name (dataset, agg, pre_agg, alpha, attack, f, const) : 
    experiment_id = f"{attack}_" 
    experiment_id += f"{agg}_" 
    experiment_id += f"{pre_agg}_" 
    if agg == "HullGuard" : 
        experiment_id += "random_"
    experiment_id += f"{const["criterion_name"]}_" 
    experiment_id += f"{dataset}_" 
    experiment_id += f"{f}_" 
    experiment_id += f"{alpha}_" 
    experiment_id += f"{const["n_workers"]}_" 
    experiment_id += f"{const["batch_size"]}_" 
    experiment_id += f"{const["beta"]}" 
    experiment_id += '_filter_score.pt'

    return experiment_id



def get_pandas_results () : 
    experiment_folder = 'test_seed1'

    constant_parameters = {
                    'n_workers': 60,
                    'batch_size': 32,
                    'beta': 0.9,
                    'criterion_name': 'CrossEntropy'
                }


    variable_parameters = {
                        "attack_name": ["ALIE","FOE","Mimic","MinSum","LF"],
                        "aggregator_name": ["CWMed","CwTM","RFA", "GAS", "HullGuard"],
                        "pre_aggregator_name": ["BKT"],
                        "n_byzantine_workers" : [8, 14, 20, 26],
                        "alpha" : [10, 1, 0.1], 
                        'dataset_name': ["CIFAR10", "MNIST", "Fashion_MNIST", "EuroSAT"]
                    }
    

    index = [[], [], [], []]

    for dataset in variable_parameters["dataset_name"] :
        for alpha in variable_parameters["alpha"] :
            for agg in variable_parameters["aggregator_name"]: 
                for pre_agg in variable_parameters["pre_aggregator_name"] :
                    if agg == "HullGuard" : 
                        pre_agg = "BKT"
                    index[0].append(dataset)
                    index[1].append(alpha)
                    index[2].append(agg)
                    index[3].append(pre_agg) 

    columns = [[], []]
    
    for attack in variable_parameters["attack_name"] : 
        for f in variable_parameters["n_byzantine_workers"] :
            columns[0].append(attack)
            columns[1].append(f) 


    values = [] 
    for dataset in variable_parameters["dataset_name"] :
        for alpha in variable_parameters["alpha"] : 
            for agg in variable_parameters["aggregator_name"]: 
                for pre_agg in variable_parameters["pre_aggregator_name"] :
                    # 1 ligne du tableau 
                    row = []
                    for attack in variable_parameters["attack_name"] : 
                        for f in variable_parameters["n_byzantine_workers"] : 
                            if agg == "HullGuard" : 
                                pre_agg = "None"
                                if f == 22 : 
                                    f = 20
                            file_name = get_file_name(dataset, agg, pre_agg, 
                                                      alpha, attack, f, 
                                                      constant_parameters)
                            path = join(experiment_folder, file_name) 
                            acc = get_mean_accuracy(path) 
                            row.append(acc) 
                    values.append(row) 


    df = pd.DataFrame(values, index=index, columns=columns)
    print(df)
    return df 


def get_hullguard_filter_result () : 
    experiment_folder = 'test_seed1'

    constant_parameters = {
                    'n_workers': 60,
                    'batch_size': 32,
                    'beta': 0.9,
                    'criterion_name': 'CrossEntropy',
                    "aggregator_name": "HullGuard",
                    "pre_aggregator_name": "None"
                }


    variable_parameters = {
                        "attack_name": ["ALIE","FOE","Mimic","MinSum","LF"],
                        "n_byzantine_workers" : [8, 14, 20, 26],
                        "alpha" : [10, 1, 0.1], 
                        'dataset_name': ["CIFAR10", "MNIST", "Fashion_MNIST", "EuroSAT"]
                    }
    

    index = [[], []]

    for dataset in variable_parameters["dataset_name"] :
        for alpha in variable_parameters["alpha"] :
            index[0].append(dataset)
            index[1].append(alpha)

    columns = [[], []]
    
    for attack in variable_parameters["attack_name"] : 
        for f in variable_parameters["n_byzantine_workers"] :
            columns[0].append(attack)
            columns[1].append(f) 


    values = [] 
    agg = constant_parameters["aggregator_name"]
    pre_agg = constant_parameters["pre_aggregator_name"]
    for dataset in variable_parameters["dataset_name"] :
        for alpha in variable_parameters["alpha"] : 
            # 1 ligne du tableau 
            row = []
            for attack in variable_parameters["attack_name"] : 
                for f in variable_parameters["n_byzantine_workers"] : 
                    file_name = get_filter_file_name(dataset, agg, pre_agg, 
                                                alpha, attack, f, 
                                                constant_parameters)
                
                    path = join(experiment_folder, file_name) 
                    filter_score = get_mean_filter_score(path, constant_parameters['n_workers'], f) 
                    row.append(filter_score) 
            values.append(row) 


    df = pd.DataFrame(values, index=index, columns=columns)
    print(df)
    return df 





if __name__ == "__main__" : 
    get_pandas_results() 

    