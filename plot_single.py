import torch 
import matplotlib.pyplot as plt 
import pandas as pd 
from os.path import join 


def load_data (path, feature_name) : 
    data = torch.load(path) 
    return data[feature_name]


def plot_feature (path, feature_name, legend, color_list, save_path, ymin=0, ymax=1) : 

    data = load_data(path, feature_name) 


    plt.figure(figsize=(6, 4))

    plt.plot(data, color=color_list[0], label=legend) 
    plt.legend() 
    plt.xlabel("Evaluation Step") 
    plt.ylabel(feature_name)
    plt.ylim((ymin, ymax))
    
    plt.savefig(save_path)

colors = ['#87CEEB', '#B47ECF',  '#ECD266', '#F4A460', '#A8D8B9'] 
save_path = "./test.png"

path = "./test_hullguard_2c_grad/ALIE_HullGuard_None_CrossEntropy_CIFAR10_0_0.1_60_32_0.9_60_statistics.pt"
plot_feature(path, 'Accuracy', "size of sub-subset : 2*c", colors, save_path) 