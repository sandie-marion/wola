import random
import torch
from math import modf 
import numpy as np 
from torchvision import datasets, transforms
from torch.utils.data import Sampler


class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, distrib):
        self.labels = labels
        self.batch_size = batch_size
        self.distrib = distrib
        self.n_classes = len(distrib)

        # group indices per class
        self.class_indices = {c: [] for c in range(self.n_classes)}
        for idx, y in enumerate(labels):
            y = int(y) 
            self.class_indices[y].append(idx)

        # compute number per class
        self.per_class = self.find_sample_repartition(distrib) 

        for idx, n in enumerate(self.per_class) : 
            if len(self.class_indices[idx]) > 0 : 
                while n > len(self.class_indices[idx]) : 
                    self.class_indices[idx] += self.class_indices[idx]


    def find_sample_repartition (self, distrib) : 
        item_per_class = []
        frac_part = [] 

        for c in range(self.n_classes):
            frac, n = modf(self.batch_size*distrib[c])
            item_per_class.append(int(n))
            frac_part.append(frac)
        

        remain = int(self.batch_size - sum(item_per_class)) 
        array = np.array(frac_part) 
        args = array.argsort()[::-1]

        for i in range (remain) :
            item_per_class[args[i]] += 1 

        return item_per_class 
        

    def __iter__(self):
        while True:
            batch = []
            for c in range(self.n_classes):
                if len(self.class_indices[c]) > 0 : 
                    batch += random.sample(
                        self.class_indices[c],
                        self.per_class[c]
                    )
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.labels) // self.batch_size