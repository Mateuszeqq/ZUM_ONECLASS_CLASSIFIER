from src.data.process_data import get_data
import pandas as pd
import numpy as np
from src.models import OneClassClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import argparse


def main(cli_args):
    
    X = get_data(cli_args.dataset_id)
    
    model_type = cli_args.model_type

    X_normal = X[X["target"] == 1]
    X_anomaly = X[X["target"] == 0]

    X_normal_noise = X_anomaly.sample(frac=cli_args.noise)
    X_normal_noise = pd.concat([X_normal, X_normal_noise])

    y_anomaly = X_anomaly["target"]
    X_normal_noise.drop(columns=["target"], inplace=True)
    X_anomaly.drop(columns=["target"], inplace=True)

    if model_type == 'if':
        model = IsolationForest(random_state=42)
        model.fit(X_normal_noise.values)

        y_pred = model.predict(X_anomaly.values)
        y_pred = np.where(y_pred == -1, 0, 1)

    elif model_type == 'svm':
        model = OneClassSVM(gamma='auto', kernel="rbf")
        model.fit(X_normal_noise.values)
        y_pred = model.predict(X_anomaly.values)
        y_pred = np.where(y_pred == -1, 0, 1)
    else:
        one_class_classifier = OneClassClassifier(
        n_of_splits = cli_args.splits,
        df = X_normal_noise,
        model_type=model_type
        )
        one_class_classifier.train_models()

        y_pred = model.predict(X_anomaly.values)

    output_df = pd.DataFrame({
        'target': y_anomaly,
        'preds': y_pred,
                           })
    
    output_df.to_csv(f"results_{model_type}_{cli_args.splits}.csv")

    print(f'Fraction of anomalies: {(len(y_pred) - sum(y_pred)) / len(y_pred)}')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', required=True, default=52, choices=[52, 94], type=int)
    parser.add_argument('--model_type', required=True, default='nn', choices=['nn', 'svm', 'if', 'rf'], type=str)
    parser.add_argument('--noise', required=False, default=0.16, type=float)
    parser.add_argument('--splits', required=False, default=3, type=int)


    cli_args = parser.parse_args()
    main(cli_args)
    

