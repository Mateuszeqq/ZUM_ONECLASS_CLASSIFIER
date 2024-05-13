import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
from src.models.MLP import Perceptron


class OneClassClassifier():
    def __init__(self, df, n_of_splits, model_type="nn", batch_size=32):
        self.n_of_splits = n_of_splits
        self.df = df
        self.subsets = np.array_split(df, n_of_splits)
        self.input_dim = df.shape[1]
        self.batch_size = batch_size

        if model_type == "nn":
            self.models = [Perceptron(self.input_dim) for _ in range(n_of_splits)]
        elif model_type == "rf":
            self.models = []
        else:
            raise("Invalid model param!")
        self.model_type = model_type

    def prepare_data_nn(self, idx):
        subsets = copy.deepcopy(self.subsets)
        for i, subset in enumerate(subsets):
            if i == idx:
                subset["target"] = 0
            else:
                subset["target"] = 1
        combined_df = pd.concat(subsets, ignore_index=True)

        features = torch.Tensor(combined_df.drop('target', axis=1).values)
        targets = torch.Tensor(combined_df["target"].values)

        dataset = TensorDataset(features, targets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
    
    def prepare_data_rf(self, idx):
        subsets = copy.deepcopy(self.subsets)
        for i, subset in enumerate(subsets):
            if i == idx:
                subset["target"] = 0
            else:
                subset["target"] = 1
        combined_df = pd.concat(subsets, ignore_index=True)

        # Return features and targets
        return combined_df.drop('target', axis=1).values, torch.Tensor(combined_df["target"].values)

    def train_models(self, verbose=False):
        if self.model_type == "nn":
            for idx, model in enumerate(self.models):
                model.train()
                dataloader = self.prepare_data_nn(idx)
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
                for epoch in range(10):
                    for inputs, labels in dataloader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.unsqueeze(1))
                        loss.backward()
                        optimizer.step()
                    if verbose:
                        print(f'Model {idx+1}: Epoch {epoch+1}, Loss: {loss.item():.4f}')
        if self.model_type == "rf":
            for idx in range(self.n_of_splits):
                X, y = self.prepare_data_rf(idx)
                clf = RandomForestClassifier(max_depth=4, random_state=0)
                clf.fit(X, y)
                self.models.append(clf)
                if verbose:
                    print(f"Model {idx+1} is ready (RandomForest)")
        
    def predict(self, input_samples, threshold=0.5):
        if self.model_type == 'nn':
            for model in self.models:
                model.eval()

        output_preds = []
        for sample in input_samples:
            predictions = [model.predict_proba(sample) for model in self.models]

            average_prediction = torch.mean(torch.stack(predictions), dim=0)
            predicted_class = (average_prediction > threshold).int()

            output_preds.append(predicted_class.item())
        # Return avg_probability, predicted_class based on threshold
        return output_preds
    
    def full_predict(self, input_sample, threshold=0.5):
        if self.model_type == 'nn':
            for model in self.models:
                model.eval()
        predictions = [model.predict_proba(input_sample) for model in self.models]

        average_prediction = torch.mean(torch.stack(predictions), dim=0)
        predicted_class = (average_prediction > threshold).int()

        # Return avg_probability, predicted_class based on threshold
        return average_prediction, predicted_class.item()
