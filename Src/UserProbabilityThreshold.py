import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, fbeta_score, roc_curve, PrecisionRecallDisplay


def plot_precision_recall_curve(Model, x_data, y_data):
    """
    Plots the precision-recall curve for a given model and dataset.

    Parameters:
    Model (sklearn estimator): The trained model to evaluate.
    x_data (array-like): The input data to predict.
    y_data (array-like): The true labels for the input data.

    Returns:
    None: This function plots the precision-recall curve.
    """
    # predict_proba method
    y_proba = Model.predict_proba(x_data)[:, 1]

    # calculate precision and recall
    precisions, recalls, thresholds = precision_recall_curve(y_data, y_proba)
    
    # plot the precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], label='Precision', color='blue')
    plt.plot(thresholds, recalls[:-1], label='Recall', color='green')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()



def threshold_fbata(Model, x_data, y_data, beta=1, flag=False):
    """
    Finds the best threshold for the given model based on the F-beta score.

    Parameters:
    Model (sklearn estimator): The trained model to evaluate.
    X_data (array-like): The input data to predict.
    y_data (array-like): The true labels for the input data.
    beta (float): The beta value for the F-beta score. Default is 1 (F1 score).
    flag (bool): If True, returns the best threshold. If False, returns None.

    Returns:
    float or None: The best threshold if flag is True, otherwise None.
    """
    # define thresholds to evaluate
    thresholds = np.linspace(0.1, 0.9, 9)
    best_f2 = 0
    best_thresh = 0.5

    # predict_proba method
    y_probs = Model.predict_proba(x_data)[:, 1]
    # loop through thresholds    
    for t in thresholds:
        # convert probabilities to binary predictions
        preds = (y_probs >= t).astype(int)
        # calculate F-beta score
        f2 = fbeta_score(y_data, preds, beta=beta)
        # update best threshold if current is better
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = t
    # print the best threshold and F-beta score
    print(f"Best threshold: {best_thresh}, Best F2: {best_f2}")
    # return the best threshold if flag is True
    return best_thresh if flag else None



def threshold_youden_j(Model, x_data, y_data, flag=False):
    """
    Finds the optimal threshold for the given model based on Youden's J statistic.

    Parameters:
    Model (sklearn estimator): The trained model to evaluate.
    X_data (array-like): The input data to predict.
    y_data (array-like): The true labels for the input data.

    flag (bool): If True, returns the best threshold. If False, returns None.

    Returns:
    float or None: The best threshold if flag is True, otherwise None.
    """
    # predict_proba method
    y_probs = Model.predict_proba(x_data)[:, 1]
    
    # compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_data, y_probs)
    # calculate Youden's J statistic
    youden_j = tpr - fpr
    # find the optimal threshold
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    # print the best Youden’s J score
    print(f'Optimal Youden’s J Threshold: {optimal_threshold}')
    # return the best threshold if flag is True
    return optimal_threshold if flag else None


def threshold_precision_recall(Model, x_data, y_data, flag=False):
    """
    Determine the optimal classification threshold using the Precision-Recall curve.

    Args:
        model: Trained classifier with predict_proba() method
        x_data (array-like): Feature data
        y_data (array-like): True binary labels

    Returns:
        float: Optimal decision threshold that maximizes F1-score

    Notes:
        - Uses precision_recall_curve() from sklearn.metrics
        - Optimal threshold is where F1-score is maximized
        - Plots interactive Precision-Recall curve using matplotlib
    """
    # Generate probability predictions for positive class
    y_probs = Model.predict_proba(x_data)[:, 1]
    
    # Calculate precision-recall pairs and thresholds
    precision, recall, thresholds = precision_recall_curve(y_data, y_probs)

    # plot the curve
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title('Precision-Recall Curve')
    plt.show()
    
    # Calculate F1 scores (exclude last precision/recall values)
    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    
    # Find optimal threshold with highest F1 score
    optimal_idx = np.nanargmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # display msg
    print(f"The Optimal Threshold is: {optimal_threshold:.4f}")
    # return the best threshold if flag is True
    return optimal_threshold if flag else None

