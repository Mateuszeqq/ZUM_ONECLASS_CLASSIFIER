import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
from src.models.MLP import Perceptron
from src.constants import EVAL_THRESHOLD
from sklearn.metrics import accuracy_score



class OneClassClassifier:
    def __init__(self, train_df, n_of_splits, model_type="nn", batch_size=32):
        self.n_of_splits = n_of_splits
        self.train_df = train_df
        self.subsets = np.array_split(train_df, n_of_splits)
        self.input_dim = train_df.shape[1]
        self.batch_size = batch_size

        if model_type == "nn":
            self.models = [Perceptron(self.input_dim) for _ in range(n_of_splits)]
        elif model_type == "rf":
            self.models = []
        else:
            raise("Invalid model param!")
        self.model_type = model_type
        self.models_eval_score = []

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

    def train_models(self, X_eval, y_eval, verbose=False):
        if self.model_type == "nn":
            for idx, model in enumerate(self.models):
                model.train()
                dataloader = self.prepare_data_nn(idx)
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
                for epoch in range(25):
                    for inputs, labels in dataloader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.unsqueeze(1))
                        loss.backward()
                        optimizer.step()
                    if verbose:
                        print(f'Model {idx+1}: Epoch {epoch+1}, Loss: {loss.item():.4f}')
                model_score = self.evaluate(model, X_eval, y_eval)
                self.models_eval_score.append(model_score)

        if self.model_type == "rf":
            for idx in range(self.n_of_splits):
                X, y = self.prepare_data_rf(idx)
                model = RandomForestClassifier(max_depth=4, random_state=0)
                model.fit(X, y)
                self.models.append(model)
                if verbose:
                    print(f"Model {idx+1} is ready (RandomForest)")

                model_score = self.evaluate(model, X_eval, y_eval)
                self.models_eval_score.append(model_score)
        

    def evaluate(self, model, X_eval, y_eval):

        if self.model_type == 'nn':
            model.eval()
            features = torch.Tensor(X_eval.values)
            targets = torch.Tensor(y_eval.values)

            dataset = TensorDataset(features, targets)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            all_preds = []
            for inputs, _ in dataloader:

                output = model(inputs)
                preds = (output > EVAL_THRESHOLD).int()
                all_preds.extend(preds)

        elif self.model_type == 'rf':
            all_preds = model.predict(X_eval.values)

        return accuracy_score(y_eval.values, all_preds)

    def get_top_models(self, fraction):
        # Combine items and values Cichosz to kurwa into tuples
        combined = list(zip(self.models, self.models_eval_score))
    
        # Sort the combined list based on the values in descending order
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    
        # Calculate the number of items to select

        num_models_to_select = max(int(len(sorted_combined) * fraction), 1)
    
        # Get the top x% of items
        top_models = [item for item, value in sorted_combined[:num_models_to_select]]
    
        return top_models

    def predict(self, input_samples, best_models_fraction=0.15, threshold=0.5):
        if self.model_type == 'nn':
            for model in self.models:
                model.eval()

        top_models = self.get_top_models(best_models_fraction)

        output_preds = []
        for sample in input_samples:
            if self.model_type == 'nn':
                sample = torch.Tensor(sample).reshape(1, -1)
                predictions = [model.predict_proba(sample) for model in top_models]
                average_prediction = torch.mean(torch.stack(predictions), dim=0)
                predicted_class = (average_prediction > threshold).int().item()
            elif self.model_type == 'rf':
                sample = sample.reshape(1, -1)
                predictions = [model.predict_proba(sample) for model in top_models]
                average_prediction = np.mean(predictions, axis=0)[0]
                predicted_class = 1 if average_prediction[1] >= threshold else 0
            
            output_preds.append(predicted_class)
        # Return avg_probability, predicted_class based on threshold
        return output_preds
    
    def full_predict(self, input_samples, threshold=0.5):
        if self.model_type == 'nn':
            for model in self.models:
                model.eval()

        output_preds = []
        for sample in input_samples:
            if self.model_type == 'nn':
                sample = torch.Tensor(sample).reshape(1, -1)
                predictions = [model.predict_proba(sample) for model in self.models]
                average_prediction = torch.mean(torch.stack(predictions), dim=0)
            elif self.model_type == 'rf':
                sample = sample.reshape(1, -1)
                predictions = [model.predict_proba(sample) for model in self.models]
                average_prediction = np.mean(predictions, axis=0)
            predicted_class = (average_prediction > threshold).int()

            output_preds.append(predicted_class.item())

        # Return avg_probability, predicted_class based on threshold
        return average_prediction, predicted_class.item()
