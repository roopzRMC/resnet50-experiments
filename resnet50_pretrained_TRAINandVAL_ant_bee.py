# %%
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import EuroSAT


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# %%
os.getcwd()
# %%
resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
# %%
resnet50.eval().to(device)
# %%
class TrainAntBeeDataSet(Dataset):
  def __init__(self):
    super().__init__()
    self.examples = self._load_examples()
    self.pil_to_tensor = transforms.ToTensor()
    self.resize = transforms.Resize((225,225))

  def _load_examples(self):
    class_names = os.listdir('hymenoptera_data/train')
    class_encoder = {class_name: idx for idx, class_name in enumerate(class_names)}
    class_decoder = {idx: class_name for idx, class_name in enumerate(class_names)}
    
    examples_list = []
    for cl_name in class_names:
      example_fp = os.listdir(os.path.join('hymenoptera_data/train',cl_name))
      example_fp = [os.path.join('hymenoptera_data/train', cl_name, img_name ) for img_name in example_fp]
      example = [(img_name, class_encoder[cl_name]) for img_name in example_fp]
      examples_list.extend(example)
    
    print(examples_list)
    return examples_list

  def __getitem__(self, idx):
    img_fp, img_class = self.examples[idx]
    img = Image.open(img_fp)
    
    features = self.pil_to_tensor(img)
    features = self.resize(features)
    
    return features, img_class

  def __len__(self):
    return len(self.examples)

# %%
class ValidationAntBeeDataSet(Dataset):
  def __init__(self):
    super().__init__()
    self.examples = self._load_examples()
    self.pil_to_tensor = transforms.ToTensor()
    self.resize = transforms.Resize((225,225))

  def _load_examples(self):
    class_names = os.listdir('hymenoptera_data/val')
    class_encoder = {class_name: idx for idx, class_name in enumerate(class_names)}
    class_decoder = {idx: class_name for idx, class_name in enumerate(class_names)}
    
    examples_list = []
    for cl_name in class_names:
      example_fp = os.listdir(os.path.join('hymenoptera_data/val',cl_name))
      example_fp = [os.path.join('hymenoptera_data/val', cl_name, img_name ) for img_name in example_fp]
      example = [(img_name, class_encoder[cl_name]) for img_name in example_fp]
      examples_list.extend(example)
    
    print(examples_list)
    return examples_list

  def __getitem__(self, idx):
    img_fp, img_class = self.examples[idx]
    img = Image.open(img_fp)
    
    features = self.pil_to_tensor(img)
    features = self.resize(features)
    
    return features, img_class

  def __len__(self):
    return len(self.examples)

# %%

class AntBeeClassifier(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    self.resnet50.fc = torch.nn.Linear(2048,2)
  
  def forward(self, X):
    return F.softmax(self.resnet50(X))


# %%

# %%



# %%
def train(model,traindataloader, valdataloader, epochs):
  optimiser = torch.optim.Adam(model.parameters(), lr=0.005)
  for epoch in range(epochs):
    training_loss = 0.0
    validation_loss = 0.0
    model.to(device)
    model.train()
    tr_num_correct = 0
    tr_num_examples = 0
    for inputs, labels in traindataloader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      predictions = model(inputs)
      #print(predictions.shape)
      #print(labels.shape)
      loss = torch.nn.CrossEntropyLoss()
      loss = loss(predictions, labels)
      loss.backward()
      optimiser.step()
      optimiser.zero_grad()
      training_loss += loss.item() * inputs.size(0)
      correct = torch.eq(torch.max(F.softmax(predictions, dim=1), dim=1)[1], labels)
      tr_num_correct += torch.sum(correct).item()
      tr_num_examples += correct.shape[0]
    training_loss /= len(traindataloader.dataset)
    
    model.eval()
    val_num_correct = 0
    val_num_examples = 0
    for inputs, labels in valdataloader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      predictions = model(inputs)
      #print(predictions.shape)
      #print(labels.shape)
      loss = torch.nn.CrossEntropyLoss()
      loss = loss(predictions, labels)
      validation_loss += loss.item() * inputs.size(0)
      correct = torch.eq(torch.max(F.softmax(predictions, dim=1), dim=1)[1], labels)
      val_num_correct += torch.sum(correct).item()
      val_num_examples += correct.shape[0]
    validation_loss /= len(valdataloader.dataset)        
    print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, train_accuracy = {:.2f},val_accuracy = {:.2f} '.format(epoch, training_loss, validation_loss, tr_num_correct / tr_num_examples,
                                                                                                                             val_num_correct / val_num_examples))       
    
    
# %%

## alter last layer of resnet50 to reflect 3 classes

classifier = AntBeeClassifier()
train_dataset = TrainAntBeeDataSet()
val_dataset = ValidationAntBeeDataSet()
train_loader = DataLoader(dataset = train_dataset, batch_size=4)
val_loader = DataLoader(dataset = val_dataset, batch_size=4)
train(classifier, traindataloader= train_loader, valdataloader= val_loader, epochs=10)



# %%
