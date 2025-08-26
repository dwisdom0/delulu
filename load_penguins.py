import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class PenguinsDataset(Dataset):

  def __init__(self, data, labels):
    self.data = data
    self.labels = labels

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]


def load_penguins_datasets():
  df = pd.read_csv('data/palmer_penguins/penguins.csv')
  df = df.dropna()


  for categorical_col in ['species', 'island', 'sex']:
    df[f'{categorical_col}_enc'] = LabelEncoder().fit_transform(df[categorical_col])

  feat_cols = [
    'island_enc',
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g',
    'sex_enc',
    'year'
  ]

  label_col = 'species_enc'

  data = df[feat_cols]
  data = StandardScaler().fit_transform(data)

  data = torch.tensor(data, dtype=torch.float32)
  labels = torch.tensor(df[label_col].to_numpy(), dtype=torch.float32)

  train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.8)

  return PenguinsDataset(train_data, train_labels), PenguinsDataset(test_data, test_labels)

