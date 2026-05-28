import torch
import torch.nn as nn
from torch import Tensor
from defenses import rfa

# Class Loss
################################################################################################
################################################################################################
class Loss:
    def __init__(self, loss_name: str, worker_id: int, **kwargs):
        """
        Wrapper for different loss functions.
        """
        self.name = loss_name

        if self.name == 'CrossEntropy':
            self.loss = nn.CrossEntropyLoss()

        elif self.name =='FedLC':
            self.loss = FedLC(worker_id=worker_id,**kwargs)

        elif self.name =='DMFL':
            self.loss = DMFL(worker_id=worker_id,**kwargs)

        elif self.name == "NorthStar" : 
            self.loss = NorthStar(worker_id=worker_id, **kwargs)

        else:
            print(self.name)
            raise ValueError("Unknown loss")

    def __call__(self, outputs: Tensor, labels: Tensor) -> Tensor:
        return self.loss(outputs, labels)


        
class NorthStar:
    def __init__(
        self,
        device: torch.device,
        local_distributions: Tensor,
        Byzantine_local_distribution: Tensor,
        n_classes:int,
        worker_id: int,
        epsilon: float = 1e-3, 
        under_attack: bool = False,
    ):
        """
        Worker Label Alignment loss (WoLA).
        Computes class weights based on local vs global data imbalance.
        """
        self.device = torch.device(device)
        self.distrib = None
        self.n_classes = n_classes 
        self.loss = nn.CrossEntropyLoss() 
        

        global_count = local_distributions.sum(dim=0)
        global_total = global_count.sum()
        global_distribution = global_count / global_total
        self.distrib = list(global_distribution) 


    def __call__ (self, logits : Tensor, targets : Tensor) -> Tensor : 
        return self.loss(logits, targets) 



# FedLC
################################################################################################
class FedLC:
    def __init__(
        self,
        device: torch.device,
        local_distributions: Tensor,
        Byzantine_local_distribution: Tensor,
        n_classes:int,
        worker_id: int,
        tau: float = 1e-1
    ):
        honest_and_Byzantine_distributions = torch.cat((local_distributions, Byzantine_local_distribution), dim=0)
        self.local_count = honest_and_Byzantine_distributions[worker_id].to(device)
        self.tau = tau

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Logits shape
        batch_size, C = logits.shape
    
        # Compute calibrated logits
        calibration_term = self.tau * torch.pow(self.local_count, -1 / 4)
        calibration_term = calibration_term.unsqueeze(0).expand(batch_size, C)
        calibrated_logits = logits - calibration_term
        
        # Predictions
        label_predictions = calibrated_logits[torch.arange(batch_size), targets]
    
        # Loss
        log_denominator = torch.logsumexp(calibrated_logits, dim=1)
        loss = - (label_predictions - log_denominator)
     
        return loss.mean()

# DMFL
################################################################################################
class DMFL:
    def __init__(
        self,
        device: torch.device,
        local_distributions: Tensor,
        Byzantine_local_distribution: Tensor,
        worker_id: int,
        T: float = 1e-1
    ):
        honest_and_Byzantine_distributions = torch.cat((local_distributions, Byzantine_local_distribution), dim=0)
        self.local_count = honest_and_Byzantine_distributions[worker_id].to(device)
        self.T = T

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Logits shape
        batch_size, C = logits.shape
    
        # Compute calibrated logits
        calibration_term = self.T * torch.pow(self.local_count, -1 / 4)
        calibration_term = calibration_term.unsqueeze(0).expand(batch_size, C)
        calibrated_logits = logits.clone()
        calibrated_logits[torch.arange(batch_size), targets] = logits[torch.arange(batch_size), targets] - calibration_term[torch.arange(batch_size), targets]
        
        # Predictions
        label_predictions = calibrated_logits[torch.arange(batch_size), targets]
    
        # Loss
        log_denominator = torch.logsumexp(calibrated_logits, dim=1)
        loss = - (label_predictions - log_denominator)
     
        return loss.mean()