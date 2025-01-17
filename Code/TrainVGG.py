
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (DataLoader,random_split, ConcatDataset)
from collections import defaultdict
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from torchvision.models import vgg19,VGG19_Weights
from torchvision import datasets,transforms,models
from torchvision.transforms.functional import InterpolationMode
from sklearn.metrics import classification_report, accuracy_score
from pytorch_lightning import seed_everything
import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore")

def plot_and_save_training_curves(model, window_size,job_id):
    # Ensure the directory for saving plots exists
    plots_dir = os.path.join(r'/home/eitag/HW_Master/ML/HW4/Code',model._name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Create a figure and axes object with size (14, 6)
    fig, axesx = plt.subplots(1, 3, figsize=(14, 6))
    (ax1, ax21, ax22) = axesx.flatten()

    x = range(len(model._training['step_loss']))
    # Plot the line
    y = pd.Series(model._training['step_loss'])
    y_smooth = y.rolling(window=window_size, min_periods=1).mean()

    # Original Data Plot
    # ax1.plot(x, y,marker='o', color='blue')


    # Smoothed Data Plot
    ax1.plot(x, y_smooth, label=f'Smooth {window_size}', marker='o', linestyle='--', color='red')
    ax1.set_title('Training', fontsize=16)
    ax1.set_xlabel('Step', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.grid(True)
    ax1.legend()

        # Smoothed Data Plot
    y_train = pd.Series(model._training['epoch_acc'])
    x = range(len(model._training['epoch_acc']))
    ax21.plot(x, y_train, label='Train', marker='o', linestyle='--', color='red')
    ax21.set_xlabel('Epoch', fontsize=14)
    ax21.set_ylabel('Accuracy', fontsize=14)


    y_train = pd.Series(model._validation['val_acc'])
    x = range(len(model._validation['val_acc']))
    ax21.plot(x, y_train, label='Validation', marker='o', linestyle='--', color='blue')
    ax21.set_title(f'Accuracy', fontsize=16)

    ax21.grid(True)
    ax21.legend()

    y_train = pd.Series(model._training['epoch_loss'])
    x = range(len(model._training['epoch_loss']))
    ax22.plot(x, y_train, label='Train', marker='o', linestyle='--', color='red')

    y_train = pd.Series(model._validation['val_loss'])
    x = range(len(model._validation['val_loss']))
    ax22.plot(x, y_train, label='Validation', marker='o', linestyle='--', color='blue')
    ax22.set_title(f'Cross Entropy Loss', fontsize=16)
    ax22.set_xlabel('Epoch', fontsize=14)
    ax22.set_ylabel('Loss', fontsize=14)
    ax22.grid(True)
    ax22.legend()
    # Adjust layout for better spacing
    plt.tight_layout()

    # Save plots with meaningful filenames
    # original_plot_path = os.path.join(plots_dir, 'original_data_plot.png')
    smoothed_plot_path = os.path.join(plots_dir, f'smoothed_{job_id}_{window_size}_data_plot.png')

    # fig.savefig(original_plot_path)  # Save the entire figure
    # print(f"Original data plot saved to {original_plot_path}")

    # Save only the smoothed plot as a separate file (optional, if desired)
    fig.savefig(smoothed_plot_path)
    print(f"Smoothed data plot saved to {smoothed_plot_path}")
    plt.show()
    # Close the plot to free memory
    plt.close(fig)

def plot_and_save_validation_curves(model, window_size):
    # Ensure the directory for saving plots exists
    plots_dir = os.path.join('/home/eitag/HW_Master/ML/HW4/Code',model._name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Create a figure and axes object with size (14, 6)
    fig, axesx = plt.subplots(1, 2, figsize=(14, 6))
    (ax21, ax22) = axesx.flatten()



        # Smoothed Data Plot
    y_train = pd.Series(model._validation['val_acc'])
    x = range(len(model._validation['val_acc']))
    ax21.plot(x, y_train, label='acc', marker='o', linestyle='--', color='red')
    ax21.set_title(f'Validation Accuracy', fontsize=16)
    ax21.set_xlabel('x', fontsize=14)
    ax21.set_ylabel('acc', fontsize=14)
    ax21.grid(True)
    ax21.legend()

    y_train = pd.Series(model._validation['val_loss'])
    x = range(len(model._validation['val_loss']))
    ax22.plot(x, y_train, label='Loss', marker='o', linestyle='--', color='red')
    ax22.set_title(f'Validation loss', fontsize=16)
    ax22.set_xlabel('x', fontsize=14)
    ax22.set_ylabel('loss', fontsize=14)
    ax22.grid(True)
    ax22.legend()
    # Adjust layout for better spacing
    plt.tight_layout()

    # Save plots with meaningful filenames
    # original_plot_path = os.path.join(plots_dir, 'original_data_plot.png')
    smoothed_plot_path = os.path.join(plots_dir, f'smoothed_val_{window_size}_data_plot.png')

    # fig.savefig(original_plot_path)  # Save the entire figure
    # print(f"Original data plot saved to {original_plot_path}")

    # Save only the smoothed plot as a separate file (optional, if desired)
    fig.savefig(smoothed_plot_path)
    print(f"Smoothed data plot saved to {smoothed_plot_path}")
    plt.show()
    # Close the plot to free memory
    plt.close(fig)

def split_tain_validation_test_data_loaders(train_size,validation_size,test_size,tranfrom=None,train_conf={},val_conf={},test_conf={}):
    train_ds = datasets.Flowers102(root='/home/eitag/HW_Master/ML/HW4/Data',download=True,split='train',transform=ToTensor() if not tranfrom else tranfrom)
    val_ds = datasets.Flowers102(root='/home/eitag/HW_Master/ML/HW4/Data',download=True,split='val',transform=ToTensor() if not tranfrom else tranfrom)
    test_ds = datasets.Flowers102(root='/home/eitag/HW_Master/ML/HW4/Data',download=True,split='test',transform=ToTensor() if not tranfrom else tranfrom)
    total_size = len(train_ds) + len(val_ds) + len(test_ds)
 

    single_dataset = ConcatDataset([train_ds, val_ds, test_ds])
    total_size = len(single_dataset)
    
    train_size = int(total_size * train_size)
    val_size = int(total_size * validation_size)
    test_size = total_size - train_size - val_size  # Remaining for the test set    print(f"Total number of examples are: {tot_size}")
    
    train_ds, val_ds, test_ds = random_split(single_dataset,[train_size,val_size,test_size])
    train_loader = DataLoader(train_ds,**train_conf)
    val_loader = DataLoader(val_ds,**val_conf)
    test_loader = DataLoader(test_ds,**test_conf)
    print("Train size: {:>10.3f} |\nValidation Size: {:>7.3f} |\nTest Size:{:>12.3f} |".format(
        (len(train_loader)),
        (len(val_loader)),
        (len(test_loader))
    ))
    return train_loader, val_loader, test_loader

class PyTorchModelWrapper:
    def __init__(self, model, criterion=None, optimizer=None, device=None,log_each_steps=15,name='test',lr=1e-4):
        """
        Initialize the model wrapper.

        Args:
            model (nn.Module): The PyTorch model.
            criterion (callable, optional): Loss function. Default is nn.CrossEntropyLoss.
            optimizer (torch.optim.Optimizer, optional): Optimizer. Default is Adam with lr=0.001.
            device (torch.device, optional): Device to use (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.AdamW(self.model.parameters(), lr=lr,weight_decay=1e-4)
        self._training = defaultdict(list)
        self._validation = defaultdict(list)
        self._testing = defaultdict(list)
        self._log_each_steps = log_each_steps
        self._name = name
        self.sftmx = torch.nn.Softmax(dim=1)
    def train(self, train_loader):
        """
        Train the model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
        """
        self.model.train()
        correct = []
        true_labels = []
        running_loss = 0.0
        step_counter = 0
        for inputs, targets in tqdm(train_loader,desc='Training',leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # This section is responsible for logginng the training data
            step_loss = loss.detach().item()
            running_loss += step_loss # Total epoc loss 
            if step_counter == self._log_each_steps: # log each N steps
                self._training['step_loss'] += [step_loss]
                step_counter = 0
            step_counter += 1
            outputs = self.sftmx(outputs)
            _, predicted = torch.max(outputs, 1)
            correct += predicted.detach().cpu()
            true_labels += targets.detach().cpu()


        correct = torch.stack(correct).numpy()
        true_labels = torch.stack(true_labels).numpy()
        acc = accuracy_score(true_labels,correct)

        self._training['epoch_loss'] += [running_loss / len(train_loader)]
        self._training['epoch_acc'] += [acc]
        
    def evaluate(self, dataloader,test_val=False):
        """
        Evaluate the model.

        Args:
            val_loader (DataLoader): DataLoader for validation or test data.

        Returns:
            float: Average loss.
            float: Accuracy.
        """
        self.model.eval()
        total_loss = 0.0
        correct = []
        total = 0
        true_labels= []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader,desc=f'Evaluation on {'Val set' if not test_val else 'Test set'}',leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Accuracy calculation
                _, predicted = torch.max(self.sftmx(outputs), 1)
                total += targets.size(0)
                correct += predicted.detach().cpu()
                true_labels += targets.detach().cpu()
        correct = torch.stack(correct).numpy()
        true_labels = torch.stack(true_labels).numpy()
        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(true_labels,correct)
        if not test_val:
            self._validation['val_acc'] += [acc]
            self._validation['val_loss'] += [avg_loss]
            print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {acc * 100:.2f}%")
        else:
            self._testing['test_acc'] += [acc]
            self._testing['test_loss'] += [avg_loss]
            print(f"Testing Loss: {avg_loss:.4f}, Accuracy: {acc * 100:.2f}%")
        # accuracy = correct / total
        
        return avg_loss, acc
    
    def save(self):
        """
        Save the model, optimizer, and internal states to a file.

        Args:
            filepath (str): Path to save the file.
        """
        plots_dir = os.path.join('/home/eitag/HW_Master/ML/HW4/Code',self._name, 'CKPT')
        os.makedirs(plots_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion': self.criterion,
            'device': self.device,
            '_training': self._training,
            '_validation': self._validation,
            '_testing': self._testing,
            '_log_each_steps': self._log_each_steps,
            '_name': self._name,
        }
        torch.save(checkpoint, plots_dir+'model_checkpoint.pth')
        print(f"Model saved to {plots_dir}")
    
    def load(self, filepath):
        """
        Load the model, optimizer, and internal states from a file.

        Args:
            filepath (str): Path to the saved file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.criterion = checkpoint['criterion']
        self.device = checkpoint['device']
        self._training = checkpoint['_training']
        self._validation = checkpoint['_validation']
        self._testing = checkpoint['_testing']
        self._log_each_steps = checkpoint['_log_each_steps']
        self._name = checkpoint['_name']
        self.model.to(self.device)
        print(f"Model loaded from {filepath}")
    


if __name__ == '__main__':
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    seed_everything(task_id + 42,workers=True)
    vgg_transforms = transforms.Compose([
        # Resize the shorter side to 256 while maintaining aspect ratio
        transforms.Resize(size=[256], interpolation=InterpolationMode.BILINEAR),
        # Crop the center to 224x224
        transforms.CenterCrop(size=[224, 224]),
        # Convert to tensor and scale values to [0.0, 1.0]
        transforms.ToTensor(),
        # Normalize using the specified mean and standard deviation
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    EPOCS = 10
    folds = {}
    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1,progress=True)
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify the classifier
    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),  # Input size is 25088 for VGG19
        nn.ReLU(),               # Activation function
        nn.Dropout(0.5),         # Optional dropout for regularization
        nn.Linear(1024, 102),    # Final output layer (e.g., for 102 classes)
    )

    train_loader, validation_loader, test_loader = split_tain_validation_test_data_loaders(0.5,0.25,0.25,
                                                                                        vgg_transforms,
                                                                                        train_conf={'batch_size':64,'shuffle':True,'num_workers':6,'pin_memory':True,'prefetch_factor':5},
                                                                                        val_conf={'batch_size':128,'shuffle':False,'num_workers':6,'pin_memory':True,'prefetch_factor':5},
                                                                                        test_conf={'batch_size':128,'shuffle':False,'num_workers':6,'pin_memory':True,'prefetch_factor':5})

    wrapper = PyTorchModelWrapper(model,name=f'VGG19_{task_id}')
    for epoch in range(EPOCS):
        print(f"EPOC: {task_id}")
        wrapper.train(train_loader)
        wrapper.evaluate(validation_loader)
        print("-"*20 + "Finish training, now testing" +"-"*20)
    wrapper.evaluate(test_loader,test_val=True)
    plot_and_save_training_curves(wrapper,10,task_id)
    plot_and_save_validation_curves(wrapper,task_id)
    wrapper.save()

