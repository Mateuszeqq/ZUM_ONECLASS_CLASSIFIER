from src.data.process_data import get_data
import pandas as pd
import numpy as np
from src.models.OneClassClassifier import OneClassClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import argparse
from src.metrics.auc_metrics import plot_roc, calculate_auc

def main(cli_args):
    
    X = get_data(cli_args.dataset_id)

    X, X_test = train_test_split(X, test_size=0.15, stratify=X['target'], random_state=2137)
    print(f"Test Dataset Class 0 -> {len(X_test[X_test['target'] == 0])}")
    print(f"Test Dataset Class 1 -> {len(X_test[X_test['target'] == 1])}")
    model_type = cli_args.model_type

    X_normal = X[X["target"] == 1]
    X_anomaly = X[X["target"] == 0]

    X_normal_noise = X_anomaly.sample(frac=cli_args.noise)
    X_normal_noise = pd.concat([X_normal, X_normal_noise])

    y_normal_noise = X_normal_noise["target"]
    X_normal_noise.drop(columns=["target"], inplace=True)
    X_anomaly.drop(columns=["target"], inplace=True)

    y_test = X_test["target"]
    X_test.drop(columns=["target"], inplace=True)

    if model_type == 'if':
        model = IsolationForest(random_state=42)
        model.fit(X_normal_noise.values)
        y_pred = model.predict(X_test.values)
        y_pred = np.where(y_pred == -1, 0, 1)
        y_proba = 1 - (-1.0) * model.decision_function(X_test.values)

    elif model_type == 'svm':
        model = OneClassSVM(gamma='auto', kernel="rbf")
        model.fit(X_normal_noise.values)
        y_pred = model.predict(X_test.values)
        y_pred = np.where(y_pred == -1, 0, 1)
        y_proba = 1 - (-1.0) * model.decision_function(X_test.values)

    else:
        X_train, X_eval, _, y_eval = train_test_split(
            X_normal_noise,
            y_normal_noise,
            test_size=cli_args.val_size,
            random_state=42,
        )

        model = OneClassClassifier(
            n_of_splits = cli_args.splits,
            train_df = X_train,
            model_type = model_type,
        )

        model.train_models(
            X_eval = X_eval,
            y_eval = y_eval
        )

        y_pred, y_proba = model.predict(X_test.values, cli_args.best_models_fraction)

    output_df = pd.DataFrame({
        'target': y_test,
        'preds': y_pred,
                           })
    
    output_df.to_csv(f"results_{model_type}_{cli_args.splits}.csv")

    print(f'Fraction of anomalies: {(len(y_pred) - sum(y_pred)) / len(y_pred)}')

    print(f"-----> y_test\n", y_test.values)
    print(f"-----> y_proba\n", y_proba)
    print(f"LENGTH OF TEST DATASET = {len(y_test)}")
    auc_score = calculate_auc(y_test, y_proba)
    plot_roc(y_test, y_proba, auc_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', required=True, default=52, choices=[52, 94], type=int)
    parser.add_argument('--model_type', required=True, default='nn', choices=['nn', 'svm', 'if', 'rf', 'xgb'], type=str)
    parser.add_argument('--noise', required=False, default=0.16, type=float)
    parser.add_argument('--splits', required=False, default=3, type=int)
    parser.add_argument('--val_size', required=False, default=0.15, type=float)
    parser.add_argument('--best_models_fraction', required=False, default=0.5, type=float)

    cli_args = parser.parse_args()
    main(cli_args)
