import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms as tfs


class Data_MNIST:
      
    def __init__(self, datadir, batch_size, num_tasks, num_cycles):

        self.datadir = datadir
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.num_cycles = num_cycles

        self._setup()

            
    def _setup(self):
        self._get_dataset()
        self._create_tasks()
        self._create_loaders()


    def _get_dataset(self):
        print("Loading MNIST dataset from {}.".format(self.datadir))
        
        self.train_dataset = datasets.MNIST(self.datadir,
                                           train=True,
                                           download=True,
                                           transform=transforms.ToTensor())
        
        self.test_dataset = datasets.MNIST(self.datadir,
                                           train=False,
                                           download=True,
                                           transform=transforms.ToTensor())


    def _create_tasks(self):
        self.train_task_datasets, self.test_task_datasets = [], []

        if self.num_tasks > 1:
            print("Splitting training and test datasets into {} parts for cl.".format(self.num_tasks))
            train_targets = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
            test_targets = [self.test_dataset[i][1] for i in range(len(self.test_dataset))]
            self.labels = np.unique(train_targets)

            err_message = "Targets are assumed to be integers from 0 up to number of classes."
            assert set(self.labels) == set(range(self.num_classes)), err_message
            err_message =  "Number of classes should be divisible by the number of tasks."
            assert self.num_classes % self.num_tasks == 0, err_message

            num_concurrent_labels = self.num_classes // self.num_tasks

            for i in range(0, self.num_classes, num_concurrent_labels):
                concurrent_labels = self.labels[i: i + num_concurrent_labels]

                trainset_filtered_indices = np.where(np.isin(train_targets, concurrent_labels))[0]
                testset_filtered_indices = np.where(np.isin(test_targets, concurrent_labels))[0]
                train_ds_subset = torch.utils.data.Subset(self.train_dataset,
                                                          trainset_filtered_indices)
                test_ds_subset = torch.utils.data.Subset(self.test_dataset,
                                                         testset_filtered_indices)
                self.train_task_datasets.append(train_ds_subset)
                self.test_task_datasets.append(test_ds_subset)
        else:
            self.labels = np.array([i for i in range(self.num_classes)])
            self.train_task_datasets = [self.train_dataset]
            self.test_task_datasets = [self.test_dataset]

        err_message = "Number of train datasets and the number of test datasets should be equal."
        assert len(self.train_task_datasets) == len(self.test_task_datasets), err_message


    def _create_loaders(self):
        self.num_classes = (self.num_classes,)
        _collate_func = default_collate

        print("Creating train and test data loaders.")
        self.train_loaders, self.test_loaders = [], []

        for ds in self.train_task_datasets:
            self.train_loaders.append(torch.utils.data.DataLoader(ds,
                                                                  batch_size=self.batch_size,
                                                                  shuffle=True,
                                                                  collate_fn=_collate_func))
        for ds in self.test_task_datasets:
            self.test_loaders.append(torch.utils.data.DataLoader(ds,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=True,
                                                                 collate_fn=_collate_func))
        self.train_loaders = self.train_loaders * self.num_cycles
        self.test_loaders = self.test_loaders * self.num_cycles

