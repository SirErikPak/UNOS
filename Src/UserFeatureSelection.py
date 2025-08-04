import pandas as pd
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif



def compute_entropy_gini_impurity(data):
    """
    Compute entropy, maximum entropy, Gini impurity, and maximum Gini impurity for each column in a DataFrame.

    Parameters:
    (pd.DataFrame): Input DataFrame with categorical or discrete numerical data.

    Returns:
        pd.DataFrame: A DataFrame containing:
        A DataFrame containing:
        - ColumnName: Name of the column
        - NumberOfCategories: Unique categories in the column
        - Entropy: Entropy of the column
        - MaxEntropy: Maximum possible entropy for the column
        - EntropyPercent: Percentage of max entropy
        - GiniImpurity: Gini impurity of the column
        - MaxGiniImpurity: Maximum possible Gini impurity for the column
        - GiniPercent: Percentage of max Gini impurity
    """
    # ititialize list
    results = []
    # iterate
    for column in data.columns:
        values, counts = np.unique(data[column].dropna(), return_counts=True)
        num_categories = len(values)
        probabilities = counts / counts.sum()

        # compute entropy and maximum entropy
        entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Avoid log(0)
        max_entropy = np.log2(num_categories) if num_categories > 1 else 0

        # compute Gini impurity and maximum Gini impurity
        gini_impurity = 1 - np.sum(probabilities**2)
        # The formula returns 0, which represents no impurity because there is no variation.
        max_gini_impurity = 1 - (1 / num_categories) if num_categories > 1 else 0

        # compute percentages
        if max_entropy != 0:
            enty_per = (entropy_value / max_entropy) * 100
        else:
            enty_per = 0  # or handle appropriately

        if max_gini_impurity != 0:
            gini_per = (gini_impurity / max_gini_impurity) * 100
        else:
            gini_per = 0  # or handle appropriately

        # append to list
        results.append([column, num_categories, entropy_value, max_entropy, enty_per, gini_impurity, max_gini_impurity, gini_per])
        #  dataframe
        resutDF = pd.DataFrame(results, columns=['ColumnName', 'NumberOfCategories', 'Entropy', 'MaxEntropy', 'EntropyPercent', 'GiniImpurity', 'MaxGiniImpurity', 'GiniPercent'])
        
    return resutDF.sort_values(by=['EntropyPercent','GiniPercent'], ascending=False)



def select_kbest_best_function(x_data, y_data, K='all'):
    """
    Selects the top features for classification tasks using multiple scoring functions
    provided by SelectKBest, and chooses the one with the best average scores.
    
    Args:
        x_data: Feature matrix (DataFrame or NumPy array).
        y_data: Target labels (Series or array-like).
        K: Number of top features to select (default: 'all').
    
    Returns:
        best_scores_df: DataFrame with selected features, scores, and p-values for the optimal scoring method.
    """
    # capture original column names if x_data is a DataFrame
    if isinstance(x_data, pd.DataFrame):
        original_feature_names = x_data.columns
    else:
        try:
            x_data = pd.DataFrame(x_data, columns=[f'Feature_{i}' for i in range(x_data.shape[1])])
            original_feature_names = x_data.columns
        except Exception as e:
            raise ValueError("x_data should be a DataFrame or convertible to one.") from e
    
    # remove constant features
    constant_filter = VarianceThreshold(threshold=0)
    x_data_filtered = constant_filter.fit_transform(x_data)
    filtered_feature_names = original_feature_names[constant_filter.get_support()]

    # Convert filtered data back to DataFrame with appropriate column names
    x_data_filtered_df = pd.DataFrame(x_data_filtered, columns=filtered_feature_names)

    # define scoring functions
    FUNCTIONS = {
        'f_classif': f_classif,
        'mutual_info_classif': mutual_info_classif
    }

    # initialize comparison variables
    best_function = None
    best_scores_df = None
    highest_avg_score = -float('inf')

    # Iterate over the functions
    for name, func in FUNCTIONS.items():
        selector = SelectKBest(score_func=func, k=K)
        selector.fit(x_data_filtered_df, y_data)

        scores = selector.scores_
        p_values = getattr(selector, 'pvalues_', [None] * len(scores))

        # create a DataFrame of scores
        feature_scores_df = pd.DataFrame({
            'Feature': filtered_feature_names,
            'Score': scores,
            'p_value': p_values
        }).sort_values(by='Score', ascending=False)

        # calculate average score for the function
        avg_score = feature_scores_df['Score'].mean()

        # update best function if the current one is better
        if avg_score > highest_avg_score:
            highest_avg_score = avg_score
            best_function = name
            best_scores_df = feature_scores_df[feature_scores_df['Feature'].isin(
                filtered_feature_names[selector.get_support()]
            )]
        
    # display function used
    print(f"Best Scoring Function: {best_function}")
        
    return best_scores_df