import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# split test and train
from sklearn.model_selection import train_test_split
# sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
# scale
from sklearn.preprocessing import MinMaxScaler
# import lfeature selection ibrary & functions
from sklearn.feature_selection import SelectKBest, f_classif, chi2, VarianceThreshold, mutual_info_classif
# import libraries for i
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from joblib import parallel_backend




def classifier_metrics(Model, y_pred, y_data, flag=None, display=True):
    """
    Classification metric for Project includes 
    Model metrics & Confusion Matrix.
    """
    # Create confusion matrix
    cm = metrics.confusion_matrix(y_data, y_pred, labels=Model.classes_)
    
    # Initialize variables
    TN, FP, FN, TP = cm.ravel()
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0  # Specificity
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity / Recall
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0  # Accuracy

    prec = TP / (TP + FP) if ( TP + FP) > 0 else 0  # Precision
    f1_score = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0  # F1 Score
    
    avg_prec = metrics.average_precision_score(y_data, y_pred)

    if display:    
    # print messages
        if flag:
            print("*" * 5 + " Classification Metrics for Validation/Test:")
        else:
            print("*" * 5 + " Classification Metrics for Training:")
            
        # classification report for more metrics
        print("Classification Report:\n", metrics.classification_report(y_data, y_pred, zero_division=0))

    # Calculate ROC curve and AUC
    fpr, tpr, _ = metrics.roc_curve(y_data, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    
    if display:
        # Plot confusion matrix and ROC curve in a single figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'], cbar=False, ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        if flag:
            ax1.set_title("Validation/Test Confusion Matrix")
        else:
            ax1.set_title("Training Confusion Matrix")

        # Plot ROC curve
        ax2.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax2.legend(loc="lower right")

        # Show the combined plot
        plt.tight_layout()
        plt.show()
    
    return spec, recall, acc, prec, f1_score, avg_prec, roc_auc


def threshold_classification(y_probs, y_data):
    # compute precision-recall curve
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_data, y_probs)

    # find index where |Precision - Recall| is minimized
    best_idx = np.argmin(np.abs(precisions[:-1] - recalls[:-1]))
    best_threshold = thresholds[best_idx]

     # plot Precision-Recall vs. Threshold
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], label="Precision", linestyle="--", color='blue')
    plt.plot(thresholds, recalls[:-1], label="Recall", color='green')
    
    # add vertical line at intersection point
    plt.axvline(x=best_threshold, color='red', linestyle='dotted', label=f"Threshold = {best_threshold:.2f}")
    
    # labels and legend
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision-Recall vs. Threshold")
    plt.legend()
    plt.show()

    return best_threshold


def classification_main(algorithm, model, desc, Model, x_data, y_data, type, train_threshold=None, metric_df=None, display=True):
    """
    This function evaluates a classification model's performance on a given dataset by calculating 
    key metrics such as specificity, sensitivity, accuracy, precision, F1 score, average precision, 
    and AUC (Area Under the Curve). It then compiles these metrics into a DataFrame, allowing users 
    to track performance results across different models.
    
    Parameters:
        algorithm (str): The algorithm name.
        model (str): The model identifier.
        Desc (str): Model description.
        Model: Trained classification model.
        x_data (pd.DataFrame): Feature data.
        y_data (pd.Series): Target labels.
        type (str): Either 'training' or 'validation/test'.
        threshold (float, int, bool, optional): Custom threshold for classification; if True, uses optimal threshold.
        metric_df (pd.DataFrame, optional): Existing DataFrame to store results.
        display (bool): Whether to display metrics.
    
    Returns:
        pd.DataFrame: Updated metrics DataFrame.
        (optional) float: Best threshold if threshold=True.
    """
    type = type.capitalize()
    flag = type != 'Training'  # Validation/Test flag
    best_threshold = None
    threshold_value = None

    # compute probabilities for thresholding
    y_probs = Model.predict_proba(x_data)[:, 1] if train_threshold is not None  else None

     # apply threshold if specified
    if train_threshold is True:
        best_threshold = threshold_classification(y_probs, y_data)
        threshold_value = best_threshold
    # apply supplied theshold value
    elif isinstance(train_threshold, (np.float32, int, float)):
        if 0 <= train_threshold <= 1:
            threshold_value = train_threshold
        else:
            raise ValueError("Threshold must be between 0 and 1.")

    # no thresholding is needed, use original ydata
    if threshold_value is not None:
        y_pred = pd.Series((y_probs > threshold_value).astype(int))
        desc  += f" - Threshold: {threshold_value:.3f}"
    else:
        y_pred = Model.predict(x_data)

    # Compute classification metrics
    metrics = classifier_metrics(Model, y_pred, y_data, flag, display)
    
    # Compile metrics into DataFrame
    df_metrics = metrics_classfication(algorithm, model, desc, type, *metrics)
    
    # Update existing DataFrame if provided
    new_df = pd.concat([metric_df, df_metrics], ignore_index=True) if metric_df is not None and not metric_df.empty else df_metrics

    return new_df, threshold_value



def stratified_grid(Model, parameters, x_data, y_data, seed, n_jobs=-1, n_split=5, score = 'roc_auc'):
    """
    Ten fold CV Stratified
    """
    # instantiate Stratified K-Fold cross-validation takes into account the class distribution
    cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed)

    # perform GridSearchCV
    GSC_estimator = GridSearchCV(Model, parameters, scoring=score, cv=cv, n_jobs=n_jobs)

    # evaluate a score by cross-validation
    scores = cross_val_score(GSC_estimator, X=x_data, y=y_data, scoring=score, cv=cv, n_jobs=n_jobs)

    # print average accuracy score CV with standard deviation
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


    with parallel_backend("threading"):
        # fit model
        fit = GSC_estimator.fit(x_data, y_data)
    
    return fit



def bayesian_optimize(Model, x_data, y_data, search_space, custom_scorer, n_jobs=-1, n_iter=64, n_splits=10, seed=42):
    """
    Perform Bayesian Optimization for hyperparameter tuning using BayesSearchCV.
    
    Args:
    - Model: The machine learning model to optimize.
    - x_data: Feature matrix.
    - y_data: Target labels.
    - search_space: Dictionary specifying the parameter search space.
    - custom_scorer: Custom scoring function.
    - n_jobs: Number of parallel jobs (default: -1, use all CPUs).
    - n_iter: Number of iterations for optimization (default: 64).
    - n_splits: Number of splits for Stratified K-Fold cross-validation (default: 10).
    - seed: Random state for reproducibility (defaul: 42)
    
    Returns:
    - Bestmodel: The best estimator after optimization.
    - BayesSearchCV object containing optimization results.
    """
    
    # ensure the search space is defined properly
    if not isinstance(search_space, dict):
        raise ValueError("Search space must be a dictionary with parameter names and ranges.")
    
    # initialize Stratified K-Fold cross-validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # perform Bayesian Optimization using BayesSearchCV
    opt = BayesSearchCV(
        estimator=Model,
        search_spaces=search_space,
        n_iter=n_iter,
        cv=kfold,
        n_jobs=n_jobs,
        scoring=custom_scorer,
        random_state=seed
    )
    # fit 
    BayesianOpt = opt.fit(x_data, y_data)
    
    # extract the best model and parameters
    Bestmodel = BayesianOpt.best_estimator_
    best_params = BayesianOpt.best_params_
    best_score = BayesianOpt.best_score_
    
    # display results
    print("Best parameters found:", best_params)
    print("Best score achieved:", best_score)
    print("\nCurrent Model Parameters:\n", Bestmodel)
    
    # Return results
    return Bestmodel


def logistic_feature_importance(Model, figsize=(8,10), fontsize=8):
    """
    This function analyzes the importance of features in a logistic regression model by processing its 
    coefficients. It creates a DataFrame with each feature's name, coefficient, effect description, 
    odds ratio, percentage change in odds, and probability, including a horizontal bar plot of feature importance.
    
    Args:
    - model: Trained logistic regression model (e.g., from sklearn).
    - figsize: Tuple defining the figure size for the plot (default: (8, 10)).
    - fontsize: Font size for axis labels and title (default: 8).
    
    Returns:
    - DataFrame: A DataFrame with feature importance details.
    """
    # Check if the model has been fitted and has the coef_ attribute
    if not hasattr(Model, 'coef_'):
        raise ValueError("The model must be a fitted logistic regression model.")
    
    # Get feature names and coefficients
    feature_names = Model.feature_names_in_
    coefficients = Model.coef_
    
    # If it's a multi-class logistic regression, handle each class separately
    if coefficients.ndim  > 1:
        coeff_list = []
        for i in range(coefficients.shape[0]):
            class_name = f"Class {i}"
            class_coefficients = coefficients[i]
            coeff_list.append(pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': class_coefficients,
                'Description': ['Decrease in the log-odds of the Positive Class' if x < 0 else 'Increase in the log-odds of the Positive Class' for x in class_coefficients],
                'Odd Ratio': np.exp(class_coefficients),
                'Percentage Change in Odds': (np.exp(class_coefficients) - 1) * 100,
                'Probability': np.exp(class_coefficients) / (1 + np.exp(class_coefficients)),
                'Class': class_name
            }))
        # Concatenate dataframes for all classes
        lr_coeff_df = pd.concat(coeff_list, ignore_index=True)
    else:
        # Single class logistic regression
        lr_coeff_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients[0],
            'Description': ['Decrease in the log-odds of the Positive Class' if x < 0 else 'Increase in the log-odds of the Positive Class' for x in coefficients[0]],
            'Odd Ratio': np.exp(coefficients[0]),
            'Percentage Change in Odds': (np.exp(coefficients[0]) - 1) * 100,
            'Probability': np.exp(coefficients[0]) / (1 + np.exp(coefficients[0])),
        })
    
    # Sort by Coefficient for better visualization
    lr_coeff_df = lr_coeff_df.sort_values(by='Coefficient', ascending=False)
    
    # Reset the index
    lr_coeff_df.reset_index(drop=True, inplace=True)
    
    # Plot feature importance (using Odds Ratio for better interpretability)
    plt.figure(figsize=figsize)
    if 'Class' in lr_coeff_df.columns:
        # Plot for multi-class case
        for class_name in lr_coeff_df['Class'].unique():
            class_df = lr_coeff_df[lr_coeff_df['Class'] == class_name]
            plt.barh(class_df['Feature'], class_df['Odd Ratio'], label=class_name)
    else:
        # Single class logistic regression
        plt.barh(lr_coeff_df['Feature'], lr_coeff_df['Odd Ratio'], color='steelblue')
    
    plt.axvline(1, color='red', linestyle='--', label="Odd Ratio = 1 (No Effect)")
    plt.xlabel("Odds Ratio")
    plt.ylabel("Features")
    plt.title("Feature Importance in Logistic Regression")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return lr_coeff_df


def plot_feature_importance(Model, x_data, figsize=(30,30), fontsize=15, display=True):
    """
    Plot feature importance from the model
    Order List & Bar Plot of Importance
    """
    # create dataframe
    data = pd.DataFrame(Model.feature_importances_ , index=x_data.columns, columns=["Feature Importance Score"])
    # print(data.sort_values("% Feature Importance", axis=0, ascending=False))

    if display:
    # bar plot
        plt.figure(figsize=figsize)
        # create a bar plot using Seaborn
        ax = sns.barplot(data=data, y=data.index, x = data['Feature Importance Score'], orient= 'h')
        ax.set_title("Feature Importance Bar Plot", fontsize = fontsize)
        # add a grid to the x-axis/
        plt.grid(axis='x', linestyle='--')
        plt.show()

    return data



def metrics_classfication(Algorithm, Model, Desc, Type, S, R, A, P, F, AP, Auc):
    """
    Pass Classfication metrics and Model Information
    """
    # initialize DataFrame
    data = pd.DataFrame(columns=['Algorithm', 'Model', 'Description', 'DataType', 'Accuracy', 'RecallSensitivity','F1Score', 'AveragePrecision', 'Precision','Specificity', 'ROC_AUC_Score'])
    # write to DataFrame
    data.loc[len(data)] = [Algorithm, Model, Desc, Type, A, R, F, AP, P, S, Auc]

    return data