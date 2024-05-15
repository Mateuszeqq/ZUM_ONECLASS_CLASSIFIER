from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def calculate_auc(y_true, y_pred):
    # Calculate AUC score
    auc_score = roc_auc_score(y_true, y_pred)
    
    # Calculate ROC curve
    return auc_score

def plot_roc(y_true, y_pred, auc_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()