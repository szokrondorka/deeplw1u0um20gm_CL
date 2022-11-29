import os

import numpy as np
import torch
#from torch.optim.lr_scheduler import MultiStepLR

from utils import save_image

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


class SupervisedTrainer(Trainer):
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

                if results_to_log is None:
                    results_to_log = batch_results.copy()
                else:
                    for metric, result in batch_results.items():
                        results_to_log[metric] += result.data

                if (self.global_iters % self.log_interval == 0):
                    self.lr_scheduler.step()
                    neptune.send_metric('learning_rate',
                                        x=self.global_iters,
                                        y=self.optimizer.param_groups[0]['lr'])

                    if self.logdir is not None:
                        save_image(batch[0][:self.batch_size, :, :, :],
                                   name='train_images',
                                   iteration=self.global_iters,
                                   filename=os.path.join(self.logdir, 'train_images.png'))

                    results_to_log = {'train_' + key: value / self.log_interval
                                      for key, value in results_to_log.items()}

                    template = ("Task {}/{}x{}\tTrain\tglobal iter: {}, batch: {}/{}, metrics:  "
                                + "".join([key + ": {:.3f}  " for key in results_to_log.keys()]))
                    print(template.format(self.current_task + 1,
                                          self.num_tasks,
                                          self.num_cycles,
                                          self.global_iters,
                                          self.iter_count,
                                          self.iters_per_task,
                                          *[item.data for item in results_to_log.values()]))

                    for metric, result in results_to_log.items():
                        neptune.send_metric(metric, x=self.global_iters, y=result)

                    results_to_log = None
                    self.test()

    def test(self):
        test_results = None
        current_test_loader = self.test_loaders[self.current_task]

        with torch.no_grad():
            for test_batch_count, test_batch in enumerate(current_test_loader, start=0):
                test_batch_results = self.test_on_batch(*test_batch)

                if test_results is None:
                    test_results = test_batch_results.copy()
                else:
                    for metric, result in test_batch_results.items():
                        test_results[metric] += result.data

        test_results = {'test_' + key: value / (test_batch_count + 1)
                        for key, value in test_results.items()}

        template = ("Task {}/{}x{}\tTest\tglobal iter: {} ({:.2f}%), metrics: "
                    + "".join([key + ": {:.3f}  " for key in test_results.keys()]))
        print(template.format(self.current_task + 1,
                              self.num_tasks,
                              self.num_cycles,
                              self.global_iters,
                              float(self.global_iters) / self.iters * 100.,
                              *[item.data for item in test_results.values()]))

        for metric, result in test_results.items():
            neptune.send_metric(metric, x=self.global_iters, y=result)
