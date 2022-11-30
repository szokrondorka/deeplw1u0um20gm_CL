import torch 
import torch.nn as nn

# Device configuration
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
# data.py

# Convolutional neural network (two convolutional layers)
# models.py

model = ConvNet(num_classes).to(device)

# Loss and optimizer
# Train the model
# Test the model
# train.py

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')


class ExperimentManager():
    def __init__(self, no_cuda=False, num_workers=2, logdir=None, prefix='',
                 datadir=os.path.expanduser('~/datasets')):
        self.logdir = logdir
        self.prefix = prefix
        self.datadir = datadir

        self.setup_environment()
        self.setup_torch()
        self.setup_trainer()

    def setup_environment(self):
        os.makedirs(self.datadir, exist_ok=True)

        if self.logdir is not None:
            self.logdir = os.path.join(self.logdir, self.prefix)
            os.makedirs(self.logdir, exist_ok=True)

    def setup_torch(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
        print("Device: {}".format(self.device))

    def setup_trainer(self):
        self.data = data.Data(self.datadir,
                              self.dataloader_kwargs)

        self.model = models.Model(self.device,
                                  self.data.input_shape,
                                  self.data.num_classes)

        self.trainer = trainers.trainer_maker(self.data.target_type,
                                              self.device,
                                              self.model.build(),
                                              self.data.batch_size,
                                              self.data.num_tasks,
                                              self.data.num_cycles,
                                              self.data.loaders,
                                              self.logdir)

    def run_experiment(self):
        self.trainer.train()
