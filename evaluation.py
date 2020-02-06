from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
import numpy as np


# Inspired from https://scikit-learn.org/stable/modules/model_evaluation.html
def create_evaluation(y_true, y_pred, classification_threshold=0.5):
    #Apply threshold to prediction
    y_pred = (y_pred > classification_threshold).astype(np.int)
    # Calculate Multilabel Confusion Matrix
    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)

    true_negative = confusion_matrix[:, 0, 0]
    true_positive = confusion_matrix[:, 1, 1]
    false_negative = confusion_matrix[:, 1, 0]
    false_positive = confusion_matrix[:, 0, 1]

    specificity = true_negative / (true_negative + false_positive)
    sensitivity = true_positive / (true_positive + false_negative)
    false_positive_rate = false_positive / (true_negative + false_positive)
    false_negative_rate = false_negative / (false_negative + true_positive)

    f1_score_inactive = f1_score(y_true[:, 0], y_pred[:, 0], average='weighted')
    f1_score_crack = f1_score(y_true[:, 1], y_pred[:, 1], average='weighted')
    mean_f1_score = (f1_score_crack + f1_score_inactive) / 2
    # mean_f1_score = f1_score(y_true, y_pred, average="weighted")
    #print(y_true.shape, y_pred.shape)
    #print("y_true", y_true)

    return specificity, sensitivity, mean_f1_score
