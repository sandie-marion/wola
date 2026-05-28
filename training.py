import torch
import torch.nn as nn
from torch.nn import Module
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from utility import Statistics, save 
from torch.utils.data import Subset, DataLoader 
from copy import deepcopy 

from gradients import model_parameters_format, gradient_dissimilarity, flatten_gradients
from time import time 

from collections import Counter 


class ClassWiseGradients :
    def __init__ (self, device, dataset, n_classes, reg_param, clip_param) :
        self.device = device 
        self.dataset = dataset 
        self.n_classes = n_classes 
        self.reg_param = reg_param 
        self.clip_param = clip_param 
        
        targets = dataset.targets
        self.n_targets = len(targets) 
        self.class_wise_gradients = [None for i in range (n_classes)] 
        self.global_class_prop = [0 for i in range (n_classes)]
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        self.indices_of_classes = [(targets == class_idx).nonzero(as_tuple=True)[0].tolist() for class_idx in range(n_classes)]


    def generate_class_wise_datasets (self, c) : 
        class_dataset = Subset (self.dataset, self.indices_of_classes[c])
        n_targets_c = len(self.indices_of_classes[c]) 

        loader = DataLoader(class_dataset, batch_size=32) 
        return loader, n_targets_c/self.n_targets 


    def get_class_wise_gradients (self, model, loss_fct) : 
        
        for c in range (self.n_classes) : 
            gradient = [torch.zeros_like(param) for param in model.parameters()] 
            model.train() 
            model.zero_grad() 

            loader, prop = self.generate_class_wise_datasets(c) 
        
            for inputs, labels in loader : 
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)    
                loss = loss_fct(outputs, labels)
                loss.backward()
            
            #extract gradient from model update
            with torch.no_grad():
                for param_idx, param in enumerate(model.parameters()):
                    if param.grad is None:
                        continue
    
                    # Get gradient (of param) from local dataset
                    grad = param.grad.clone().detach() 
                    
                    # Store gradient
                    gradient[param_idx] = grad
                gradient = flatten_gradients(gradient)
    
            self.class_wise_gradients[c] = gradient
            self.global_class_prop[c] = prop 
            

    def update_worker_gradient (self, model, worker, target_batch) : 
        counts = list(torch.bincount(target_batch, minlength=self.n_classes))
        
        final_gradient = flatten_gradients(worker.gradient)
        for c, count in enumerate(counts) : 
            if count == 0 : 
                gradient = self.class_wise_gradients[c] 
                prop = self.global_class_prop[c] 
                gradient = torch.mul(gradient.to(self.device), prop) 
                final_gradient = torch.add(final_gradient.to(self.device), gradient.to(self.device)) 
        final_gradient = model_parameters_format(final_gradient, model) 
        return final_gradient 
            

# Descent algorithm
################################################################################################
def stochastic_heavy_ball(model, workers, aggregator, attack, train_dataset, test_loader, prop_class, kwargs):
    """
    The stochastic heavy ball algorithm is described in detail in Fixing by Mixing: A Recipe for Optimal Byzantine ML under Heterogeneity.
    """
 
    n_honest_workers = kwargs['n_honest_workers']
    f = kwargs['n_byzantine_workers']
    beta = kwargs['beta']
    device = kwargs['device']
    experiment_id = kwargs['experiment_id']
    n_classes = kwargs['n_classes']
    n_step = kwargs['n_step']
    lr = kwargs['lr']
    reg_param = kwargs['reg_param']
    clip_param = kwargs['clip_param']
    experiment_folder = kwargs['experiment_folder']
    aggregator_name = kwargs['aggregator_name']

    n_workers = n_honest_workers + f 

    statistics_to_save = Statistics()
    step = 0

    worker_iters = [iter(loader) for loader in workers.loaders()]

    row_aggregated_momentum = flatten_gradients([torch.zeros_like(param) for param in model.parameters()])

    for step in range (0, n_step) :
        start = time() 
        
        for worker_id in range (0, n_workers) : 
            model.train() 
            running_loss = 0.0


            try : 
                batch = next(worker_iters[worker_id])
            except :
                worker_iters[worker_id] = iter(workers.loaders()[worker_id])
                batch = next(worker_iters[worker_id])
                
            inputs, labels = batch 
                    
            # if an honest worker
            if workers[worker_id].honest: 
                model.zero_grad()
                
                inputs, labels = inputs.to(device), labels.to(device)
         
                outputs = model(inputs)

                reg = regularization(model, reg_param)
                
                loss = workers[worker_id].compute_loss(outputs, labels) + reg
                
                loss.backward()

                clip_grad_norm(model, clip_param)
                
                running_loss += loss.item()/n_honest_workers

                workers[worker_id].compute_momentum(model, beta) #worker.gradient stores the miniWoLA update 
                
                
            # if a Byzantine worker
            else:
                # step, net, worker, inputs, labels, row_honest_gradients, device
                row_honest_gradients = workers.get_momentums(only_honest = True, row = True)
                row_bad_gradient = attack(step, model, workers[worker_id], inputs, labels, row_honest_gradients, device)
                bad_gradient = model_parameters_format(row_bad_gradient, model)
                workers[worker_id].momentum = bad_gradient

        # Update model
        with torch.no_grad():
            row_momentums = workers.get_momentums(only_honest = False, row = True)

            row_aggregated_momentum = aggregator(row_momentums)
            
            unrow_aggregated_momentum = model_parameters_format(row_aggregated_momentum, model)
            for param_idx, param in enumerate(model.parameters()):
                param -= lr(step) * unrow_aggregated_momentum[param_idx]
            
            end = time() 

            if step % 50 == 0:
                # Compute remaining statistics to save
                accuracy = evaluate_model(model, test_loader, prop_class, device)

                row_honest_momentums = workers.get_momentums(only_honest = True, row = True)
                grad_dissimilarity = gradient_dissimilarity(row_honest_momentums)

                t_epoch = end - start 

                # Stock statistics
                statistics_to_save.append('Steps', step)
                statistics_to_save.append('RunningLoss', running_loss)
                statistics_to_save.append('GradientDissimilarity_Momentums', grad_dissimilarity)
                statistics_to_save.append('Accuracy', accuracy)
                statistics_to_save.append('Time', t_epoch)

                clear_output(wait=True)
                print(f"Experiment {kwargs['experiment_id']} Progress {step}/{kwargs['n_step']}")
                #plt.plot(statistics_to_save['Accuracy'])
                #plt.title(int(np.mean(statistics_to_save['Accuracy'])))
                #plt.show()

            running_loss = 0.0 


    # Save statistics
    save(data = statistics_to_save.data, name = 'statistics', experiment_id = kwargs['experiment_id'], experiment_folder = kwargs['experiment_folder'])
    del statistics_to_save
    del inputs, labels, outputs
    
# About Training
################################################################################################
def regularization(model: Module, l2_lambda: float) -> torch.Tensor:
    """
    Compute L2 regularization term for all model parameters.
    """
    l2_reg = sum(param.norm(2) ** 2 for param in model.parameters())
    return l2_lambda * l2_reg


def clip_grad_norm(model: Module, max_norm: float) -> None:
    """
    Clips gradients to prevent exploding gradients during training.
    """
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def lr_CIFAR10_Purchase100(step):
    """
    Learning rate for CIFAR10 ans Purchase100.
    """
    if step < 200:
        return 0.1
    return 0.05


def lr_EuroSAT(step):
    """
    Learning rate for CIFAR10 ans Purchase100.
    """
    if step < 100:
        return 0.1
    elif step < 200 : 
        return 0.05
    return 0.005
    
def lr_MNIST(step):
    """
    Learning rate for MNIST and FashionMNIST dataset.
    """
    lr = 0.75
    lr /= 1+int(step/50)
    return lr

# Evaluation
################################################################################################
def evaluate_model_old(model: Module, test_loader, device: torch.device) -> float:
    """Return model accuracy (%) on test data."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def evaluate_model(model: Module, test_loader, prop_class, device: torch.device) -> float:
    """Return model accuracy (%) on test data."""
    model.eval()
    n = len(prop_class) 
    correct = [0 for i in range (n)] 
    total = [0 for i in range (n)] 

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            for l, p in zip(labels, preds) : 
                total[l] += 1 
                if l == p : 
                    correct[l] += 1

    result = 0 
    for c, t, p in zip(correct, total, prop_class) : 
        result += (c/t)*p 

    print("result evaluation :", result) 

    return result


def get_distribution (labels : list, n_labels : int) -> torch.Tensor : 
    nb_samples = len(labels) 
    label_list = [int(l) for l in labels]
    sample_per_class = Counter(label_list)
    distribution = [0 for i in range(0,n_labels)]
    for label in sample_per_class :
        distribution[label] = sample_per_class[label] / nb_samples

    return torch.tensor(distribution)  
