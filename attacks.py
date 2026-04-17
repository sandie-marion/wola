import torch
from typing import List
import math
from training import regularization, clip_grad_norm

# Attacks involving complex code
from Attacks.nearest_neighbor_poisoning import NearestNeighborPoisoning
from Attacks.poisoned_fl import PoisonedFL
from byzfl.utils.misc import check_vectors_type


# Class Attacks
################################################################################################
################################################################################################
class Attack:
    def __init__(self, attack_name, aggregator, **kwargs):
        """
        Wrapper for different Byzantine attack strategies
        """
        self.name = attack_name
        self.byzantine_row_gradient = None
        self.step = None

        # Choose which attack to use based on name
        if self.name == 'None':
            # No attack
            self.attack = lambda x: x
            
        elif  self.name == 'Mimic':
            # Mimic attack
            mimic = Mimic(aggregator, **kwargs)
            self.attack = mimic

        elif self.name =='ALIE':
            # A Little Is Enough attack
            alie = ALittleIsEnough(aggregator, **kwargs)
            self.attack = alie
            
        elif self.name == 'FOE':
            # Fall of Empires attack
            foe = FallOfEmpires(aggregator, **kwargs)
            self.attack = foe
        
        elif self.name == 'SF':
            # Sign Flipping attack
            sf = SignFlipping(**kwargs)
            self.attack = sf

        elif self.name == 'NNP':
            # Nearest Neighbor Poisoning attack
            nnp = NearestNeighborPoisoning(**kwargs)
            self.attack = nnp

        elif self.name == 'MinSum':
            # MinSum attack
            minsum = MinSum(**kwargs)
            self.attack = minsum

        elif self.name == 'MinMax':
            # MinMax attack
            minmax = MinMax(**kwargs)
            self.attack = minmax

        elif self.name == 'PoisonedFL':
            # Poisoned Federated Learning attack
            poisonedfl = PoisonedFL(**kwargs)
            self.attack = poisonedfl
        
        elif self.name == 'LF' : 
            label_flipping = LabelFlipping(**kwargs) 
            self.attack = label_flipping

        elif self.name == 'PLF' : 
            partial_label_flipping = PartialLabelFlipping(**kwargs) 
            self.attack = partial_label_flipping
            
        else:
            raise ValueError("Unknown attack")
    
    def __call__(self, step, net, worker, inputs, labels, row_honest_gradients, device) -> torch.Tensor:
        #If Attack is Label Flipping, we compute byzantine gradient for each byzantine workers (diff values) 
        if self.name == 'LF' or self.name == 'PLF' : 
            self.byzantine_row_gradient = self.attack(net, worker, inputs, labels, device) 
        # Compute Byzantine gradient once per step
        elif self.step != step:
            if self.name == 'PoisonedFL':
                # Special handling for PoisonedFL attack
                self.byzantine_row_gradient = self.attack(step, net, row_honest_gradients)
            else:
                # Apply chosen attack to honest gradients
                # for several fixed hyperparameters, 
                self.byzantine_row_gradient = self.attack(row_honest_gradients)
                # aggreger les gradients 
            # Remember current step to avoid recomputation
            self.step = step
        # Return cached malicious gradient
        return self.byzantine_row_gradient
            
# Attacks
################################################################################################
################################################################################################


# Partial Label Flipping 
################################################################################################
class PartialLabelFlipping: 
    def __init__ (self, true_label, false_label, reg_param, clip_param, beta) : 
        self.true_label = true_label
        self.false_label = false_label

        self.reg_param = reg_param
        self.clip_param = clip_param
        self.beta = beta


    def __call__(self, net, worker, inputs, labels) :
        net.train()
        net.zero_grad() 
        outputs = net(inputs) 
        labels[labels==self.true_label] = self.false_label
        reg = regularization(net, self.reg_param)
                        
        loss = worker.compute_loss(outputs, labels) + reg
        
        loss.backward()

        clip_grad_norm(net, self.clip_param)

        worker.compute_momentum(net, self.beta)

        return worker.flatten_momentum() 
    



# Label Flipping 
################################################################################################
class LabelFlipping: 
    def __init__ (self, n_classes, shift, reg_param, clip_param, beta) : 
        self.n_classes = n_classes
        self.shift = shift 
        self.reg_param = reg_param
        self.clip_param = clip_param
        self.beta = beta


    def __call__(self, net, worker, inputs, labels, device) :
        net.train()
        net.zero_grad() 
        outputs = net(inputs.to(device)) 
        flipped_labels = torch.add(labels, self.shift) 
        flipped_labels = torch.fmod(flipped_labels, self.n_classes) 
        reg = regularization(net, self.reg_param)
                        
        loss = worker.compute_loss(outputs, flipped_labels.to(device)) + reg
        
        loss.backward()

        clip_grad_norm(net, self.clip_param)

        worker.compute_momentum(net, self.beta)

        return worker.flatten_momentum() 




# Sign Flipping
################################################################################################
class SignFlipping:
    """
    Sign Flipping attack: returns the negative of the mean honest gradient.
    """
    @torch.no_grad()
    def __call__(self, row_honest_gradients: list) -> torch.Tensor:
        # Compute mean gradient across honest workers
        mean_grad = torch.stack(row_honest_gradients, dim=0).mean(dim=0)
        # Flip the sign
        return -mean_grad
        
# Mimic
################################################################################################
class Mimic:
    """
    Mimic attack: returns the update of a chosen honest worker.
    """
    def __init__(self, aggregator, f):
        self.f = f
        self.agg = aggregator 
        self.pre_agg_list = [] 
        

    
    def mimic (self, row_honest_gradients, worker_id) : 
        selected = row_honest_gradients[worker_id]
        # Return a clone 
        return selected.clone()

        
    def evaluate(self, honest_vectors, avg_honest_vector, n_honest_workers):
       
        """
        Computes the norm of the distance between the aggregated vector (including Byzantine vectors) and the average of honest vectors.
        """

        tools, honest_vectors = check_vectors_type(honest_vectors)

        #Compute Byzantine vector
        best_distance = 100000
        best_id = 0 
        for i in range (n_honest_workers) : 
                
            byzantine_vector = self.mimic(honest_vectors, i)
            byzantine_vectors = tools.array([byzantine_vector] * self.f)
    
            #Aggregate vectors with current Byzantine vectors
            vectors = tools.concatenate((honest_vectors, byzantine_vectors), axis=0)
            for pre_agg in self.pre_agg_list:
                vectors = pre_agg(vectors)
            aggregated_vector = self.agg(list(vectors))
    
            #Return distance between aggregate vector and mean of honest vectors
            distance = tools.subtract(aggregated_vector, avg_honest_vector)
            dist = tools.linalg.norm(distance).item()
            if dist > best_distance : 
                best_distance = dist
                best_id = i 
                
        return best_id 

    @torch.no_grad()
    def __call__(self, row_honest_gradients: List[torch.Tensor]) -> torch.Tensor:

        if not row_honest_gradients:
            raise ValueError("empty input")
        G = torch.stack(row_honest_gradients, dim=0)
        tools, honest_vectors = check_vectors_type(G)
        avg_honest_vector = tools.mean(honest_vectors, axis=0)
        
        best_id = self.evaluate(honest_vectors, avg_honest_vector, len(row_honest_gradients)) 
        
        return self.mimic(row_honest_gradients, best_id) 

# A Little Is Enough
################################################################################################
class ALittleIsEnough:
    """
    Attack: return mu + z_max * sigma where z_max is the Gaussian quantile.
    """
    def __init__(self, aggregator, n_workers, f):
        self.f = f 
        self.pre_agg_list = []
        self.agg = aggregator 
        self.evals = 20 
        self.delta = 10.0 
        self.ratio = 0.8
        self.start = 0.0

    
    def alie (self, row_honest_gradients, tau:float) -> torch.Tensor:
        mu = row_honest_gradients.mean(dim=0)
        sigma = row_honest_gradients.std(dim=0, unbiased=False)
        return mu + sigma * tau


    def _evaluate(self, honest_vectors, avg_honest_vector, current_tau):
       
        """
        Computes the norm of the distance between the aggregated vector (including Byzantine vectors) and the average of honest vectors.
        """

        tools, honest_vectors = check_vectors_type(honest_vectors)

        #Compute Byzantine vector
        byzantine_vector = self.alie(honest_vectors, current_tau)
        byzantine_vectors = tools.array([byzantine_vector] * self.f)

        #Aggregate vectors with current Byzantine vectors
        vectors = tools.concatenate((honest_vectors, byzantine_vectors), axis=0)
        for pre_agg in self.pre_agg_list:
            vectors = pre_agg(vectors)
        aggregated_vector = self.agg(list(vectors))

        #Return distance between aggregate vector and mean of honest vectors
        distance = tools.subtract(aggregated_vector, avg_honest_vector)
        return tools.linalg.norm(distance).item()


    
    def expansion_phase (self, honest_vectors, avg_honest_vector) : 
        best_x = self.start
        best_y = self._evaluate(honest_vectors, avg_honest_vector, best_x)
        delta = self.delta
        remaining_evals = self.evals - 1

        while remaining_evals > 0:
            prop_x = best_x + delta
            prop_y = self._evaluate(honest_vectors, avg_honest_vector, prop_x)
            remaining_evals -= 1

            if prop_y > best_y:
                # If the new value is better: Update best_x, double the step size (delta *= 2), and continue exploring.
                best_x, best_y = prop_x, prop_y
                delta *= 2
            else:
                # If the new value is worse: Stop the expansion phase and proceed to contraction.
                delta *= self.ratio
                break

        return best_x, best_y, delta, remaining_evals

    
    def contraction_phase(self, honest_vectors, avg_honest_vector, best_x, best_y, delta, remaining_evals):

        """
        Performs the contraction phase of the optimization.
        This phase refines the search by reducing the step size (delta) and searching around the current best value.
        """
        while remaining_evals > 0:
            # Continue reducing the step size until no significant improvements are found or evaluations are exhausted.
            prop_x = best_x + delta
            prop_y = self._evaluate(honest_vectors, avg_honest_vector, prop_x)
            remaining_evals -= 1

            if prop_y > best_y:
                # If a better value is found: Update best_x and best_y.
                best_x, best_y = prop_x, prop_y

            delta *= self.ratio

        return best_x

    
    @torch.no_grad()
    def __call__(self, row_honest_gradients: List[torch.Tensor]) -> torch.Tensor:
        if not row_honest_gradients:
            raise ValueError("empty input")
        G = torch.stack(row_honest_gradients, dim=0)
        tools, honest_vectors = check_vectors_type(G)
        avg_honest_vector = tools.mean(honest_vectors, axis=0)

        # Expansion Phase
        best_tau, largest_distance, delta, remaining_evals = self.expansion_phase(honest_vectors, avg_honest_vector)
        best_tau = self.contraction_phase(honest_vectors, avg_honest_vector, best_tau, largest_distance, delta, remaining_evals) 
        
        # Set the best attack factor and execute IPM
        return self.alie(honest_vectors, best_tau)

# Fall of Empires
################################################################################################
class FallOfEmpires:
    """
    Attack: return a scaled and sign-flipped version of the mean honest gradient.
    g_bad = -epsilon * mean( honest_gradients )
    """
    def __init__(self, aggregator, f):
        self.f = f 
        self.pre_agg_list = []
        self.agg = aggregator 
        self.evals = 20 
        self.delta = 10.0 
        self.ratio = 0.8
        self.start = 0.0
        

    def foe (self, row_honest_gradients, tau) : 
        mu = row_honest_gradients.mean(dim=0)
        return -tau * mu



    def _evaluate(self, honest_vectors, avg_honest_vector, current_tau):
       
        """
        Computes the norm of the distance between the aggregated vector (including Byzantine vectors) and the average of honest vectors.
        """

        tools, honest_vectors = check_vectors_type(honest_vectors)

        #Compute Byzantine vector
        byzantine_vector = self.foe(honest_vectors, current_tau)
        byzantine_vectors = tools.array([byzantine_vector] * self.f)

        #Aggregate vectors with current Byzantine vectors
        vectors = tools.concatenate((honest_vectors, byzantine_vectors), axis=0)
        for pre_agg in self.pre_agg_list:
            vectors = pre_agg(vectors)
        aggregated_vector = self.agg(list(vectors))

        #Return distance between aggregate vector and mean of honest vectors
        distance = tools.subtract(aggregated_vector, avg_honest_vector)
        return tools.linalg.norm(distance).item()


    
    def expansion_phase (self, honest_vectors, avg_honest_vector) : 
        best_x = self.start
        best_y = self._evaluate(honest_vectors, avg_honest_vector, best_x)
        delta = self.delta
        remaining_evals = self.evals - 1

        while remaining_evals > 0:
            prop_x = best_x + delta
            prop_y = self._evaluate(honest_vectors, avg_honest_vector, prop_x)
            remaining_evals -= 1

            if prop_y > best_y:
                # If the new value is better: Update best_x, double the step size (delta *= 2), and continue exploring.
                best_x, best_y = prop_x, prop_y
                delta *= 2
            else:
                # If the new value is worse: Stop the expansion phase and proceed to contraction.
                delta *= self.ratio
                break

        return best_x, best_y, delta, remaining_evals

    
    def contraction_phase(self, honest_vectors, avg_honest_vector, best_x, best_y, delta, remaining_evals):

        """
        Performs the contraction phase of the optimization.
        This phase refines the search by reducing the step size (delta) and searching around the current best value.
        """
        while remaining_evals > 0:
            # Continue reducing the step size until no significant improvements are found or evaluations are exhausted.
            prop_x = best_x + delta
            prop_y = self._evaluate(honest_vectors, avg_honest_vector, prop_x)
            remaining_evals -= 1

            if prop_y > best_y:
                # If a better value is found: Update best_x and best_y.
                best_x, best_y = prop_x, prop_y

            delta *= self.ratio

        return best_x

    
    @torch.no_grad()
    def __call__(self, row_honest_gradients: List[torch.Tensor]) -> torch.Tensor:
        if not row_honest_gradients:
            raise ValueError("empty input")
        G = torch.stack(row_honest_gradients, dim=0)
        tools, honest_vectors = check_vectors_type(G)
        avg_honest_vector = tools.mean(honest_vectors, axis=0)

        # Expansion Phase
        best_tau, largest_distance, delta, remaining_evals = self.expansion_phase(honest_vectors, avg_honest_vector)
        best_tau = self.contraction_phase(honest_vectors, avg_honest_vector, best_tau, largest_distance, delta, remaining_evals) 
        
        # Set the best attack factor and execute IPM
        return self.foe(honest_vectors, best_tau)

        

# Min-Max
################################################################################################
class MinMax:
    @torch.no_grad()
    def __call__(self, row_honest_gradients: List[torch.Tensor]) -> torch.Tensor:
        G = torch.stack(row_honest_gradients, dim=0)   # (n, d)
        device = G.device
    
        mu = G.mean(dim=0)                             # (d,)
        perturbation = -mu / mu.norm()                 # unit direction
    
        # max pairwise honest distance
        dists = torch.cdist(G, G)                      # (n, n)
        honest_distance = dists.max()
    
        gamma = torch.tensor(50.0, device=device)
        threshold_diff = 1e-5
        gamma_fail = gamma
        gamma_succ = torch.tensor(0.0, device=device)
    
        while torch.abs(gamma_succ - gamma) > threshold_diff:
            byzantine_update = mu - gamma * perturbation
            dists = torch.norm(byzantine_update - G, dim=1)
            honest_byzantine_dist = dists.max()
    
            if honest_byzantine_dist <= honest_distance:
                gamma_succ = gamma
                gamma = gamma + gamma_fail / 2
            else:
                gamma = gamma - gamma_fail / 2
    
            gamma_fail = gamma_fail / 2
    
        byzantine_update = mu - gamma_succ * perturbation
        return byzantine_update

# Min-Sum
################################################################################################
class MinSum:
    @torch.no_grad()
    def __call__(self, row_honest_gradients: List[torch.Tensor]) -> torch.Tensor:
        G = torch.stack(row_honest_gradients, dim=0)   # (n, d)
        device = G.device

        mu = G.mean(dim=0)                             # (d,)
        perturbation = -mu / mu.norm()                 # unit direction

        dists_hh_sq = torch.cdist(G, G, p=2).pow(2)    # (n, n)
        honest_distance = dists_hh_sq.sum(dim=1).max() # scalar

        gamma = torch.tensor(50.0, device=device)
        threshold_diff = 1e-5
        gamma_fail = gamma.clone()
        gamma_succ = torch.tensor(0.0, device=device)

        while torch.abs(gamma_succ - gamma) > threshold_diff:
            byzantine_update = mu - gamma * perturbation

            dists_sq = (byzantine_update.unsqueeze(0) - G).pow(2).sum(dim=1)  # (n,)
            honest_byzantine_dist = dists_sq.sum()

            if honest_byzantine_dist <= honest_distance:
                gamma_succ = gamma
                gamma = gamma + gamma_fail / 2
            else:
                gamma = gamma - gamma_fail / 2

            gamma_fail = gamma_fail / 2

        byzantine_update = mu - gamma_succ * perturbation
        return byzantine_update