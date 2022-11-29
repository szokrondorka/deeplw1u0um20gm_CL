import os

import numpy as np
import torch
#from torch.optim.lr_scheduler import MultiStepLR

from utils import save_image

def trainer_maker(trainer_type, *args):
    return trainTutorial(*args)


class Trainer:
    def __init__(self, device, model, batch_size, num_tasks, num_cycles, data_loaders, logdir,
                 log_interval=100, iters=1000, lr=0.1, wd=5e-4, optimizer=torch.optim.SGD):
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.num_cycles = num_cycles
        self.train_loaders = data_loaders['train_loaders']
        self.test_loaders = data_loaders['test_loaders']
        self.log_interval = log_interval
        self.iters = iters

        self.optimizer = optimizer(self.model.parameters(),
                                       lr,
                                       weight_decay=wd)
        
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        self.logdir = logdir

        
# Train the model

class trainTutorial(Trainer):
    def __init__(self, *args):
        super().__init__(*args)
    
    def test_on_batch(self, input_images, target):
        input_images = input_images.to(self.device)
        target = target.to(self.device)
        model_output = self.model(input_images)
        loss_per_sample = self.loss_function(model_output, target)
        loss_mean = torch.mean(loss_per_sample)
        predictions = torch.argmax(model_output, dim=1)
        accuracy = torch.mean(torch.eq(predictions, target).float())
        results = {'total_loss_mean': loss_mean,
                   'accuracy': accuracy}
        return results

    def train_on_batch(self, batch):
        self.optimizer.zero_grad()
        results = self.test_on_batch(*batch)
        results['total_loss_mean'].backward()
        self.optimizer.step()
        return results
    
    def train(self):
        self.global_iters = 0
        self.iters_per_task = self.iters // self.num_tasks // self.num_cycles
        print("Start training.")
        
        for self.current_task in range(0, self.num_tasks * self.num_cycles):
            current_train_loader = self.train_loaders[self.current_task]
            current_train_loader_iterator = iter(current_train_loader)
            results_to_log = None

            for self.iter_count in range(1, self.iters_per_task + 1):
                self.global_iters += 1

                try:
                    batch = next(current_train_loader_iterator)
                except StopIteration:
                    current_train_loader_iterator = iter(current_train_loader)
                    batch = next(current_train_loader_iterator)

                batch_results = self.train_on_batch(batch)
    

        
