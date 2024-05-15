from torch import nn
from src.constants import HIDDEN_LAYERS_SIZE
import torch


class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_LAYERS_SIZE['first_linear']),
            nn.BatchNorm1d(HIDDEN_LAYERS_SIZE['first_linear']),
            nn.ReLU(),

            nn.Linear(HIDDEN_LAYERS_SIZE['first_linear'], HIDDEN_LAYERS_SIZE['second_linear']),
            nn.BatchNorm1d(HIDDEN_LAYERS_SIZE['second_linear']),
            nn.ReLU(),

            nn.Linear(HIDDEN_LAYERS_SIZE['second_linear'], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def predict_proba(self, x):
        # to match function predict_proba in scikit-learn Random Forest model
        return self.model(x)
