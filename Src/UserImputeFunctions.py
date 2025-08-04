from sklearn.mixture import GaussianMixture
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def gaussian_mixture_binning(data, colum_list, seed, n_init=10):
    """
    This function is designed to fit a Gaussian Mixture Model (GMM) with different numbers of 
    components (clusters) and use information criteria (AIC and BIC) to determine the optimal 
    number of components. It then visualizes the results using a plot to help identify the best 
    number of components for the GMM.
    """
    # initialize fit GMM with different number of components and select the best using AIC or BIC
    aic = [] # AIC (Akaike Information Criterion)    Lower the Better
    bic = [] # BIC (Bayesian Information Criterion)  Lower the Better
    components_range = range(1, 11)  # 1 to 10 components
    # remove any NaNs
    data = data[colum_list].dropna()
    
    for n in components_range:
        gmm = GaussianMixture(n_components=n, n_init=n_init, random_state=seed)
        gmm.fit(data[colum_list])
        aic.append(gmm.aic(data[colum_list]))
        bic.append(gmm.bic(data[colum_list]))
    
    # plot AIC and BIC to find the optimal number of components
    plt.plot(components_range, aic, label='AIC')
    plt.plot(components_range, bic, label='BIC')
    plt.xlabel('Number of Components')
    plt.ylabel('AIC/BIC')
    plt.legend()
    plt.title('AIC and BIC for GMM')
    plt.show()




def impute_gaussian_mixture(data_series, seed, n_components):
    """
    Imputes missing values in a data series using a Gaussian Mixture Model (GMM).
    """
    # apply Min-Max Scaling
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_series), columns=data_series.columns)    
    
    # Convert series to numpy array
    data = np.asarray(data_scaled)

    # Fit GMM to the observed data
    gmm = GaussianMixture(n_components=n_components, random_state=seed)
    observed_data = data[~np.isnan(data)].reshape(-1, 1)
    gmm.fit(observed_data)

    # Impute missing values by sampling from the GMM
    missing_mask = np.isnan(data)
    imputed_values, _ = gmm.sample(np.sum(missing_mask))

    # Clip the imputed values to stay within the range: MinMax Scale
    imputed_values = np.clip(imputed_values, 0, 1)

    # Fill missing data with the imputed values
    data[missing_mask] = imputed_values.flatten()

    # reverse scaling to original range
    imputed_data = pd.DataFrame(scaler.inverse_transform(data), columns=data_scaled.columns)

    # Return imputed DataFrame  
    return imputed_data



def find_gaussian_mixture_components(data, seed, component_range=range(1, 10), figsize=(8,6)):
    """
    Determines the optimal number of components for a Gaussian Mixture Model (GMM)
    using Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    Notes:
    -----
    The function fits GMMs with different numbers of components to the provided data,
    computes their AIC and BIC scores, and plots these scores to help visualize the optimal
    number of components. The optimal number is determined based on the lowest BIC score,
    as BIC tends to favor models that generalize well to unseen data by penalizing complexity
    more strongly than AIC.
    """
    # Ensure data is in numpy array format
    if hasattr(data, 'values'):
        dataArr = data.values

    # Remove NaNs
    dataArr = dataArr[~np.isnan(dataArr).any(axis=1)]

    # apply Min-Max Scaling
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(dataArr), columns=data.columns)
    
    # Initialize lists to store AIC and BIC scores
    aic_scores = []
    bic_scores = []

    for n_components in component_range:
        # Initialize the GMM
        gmm = GaussianMixture(n_components=n_components, random_state=seed)

        # Fit the GMM to the data
        gmm.fit(data_scaled)

        # Compute AIC and BIC
        aic_scores.append(gmm.aic(data_scaled))
        bic_scores.append(gmm.bic(data_scaled))

    # Plot AIC and BIC scores
    plt.figure(figsize=figsize)
    plt.plot(component_range, aic_scores, label='AIC', marker='o')
    plt.plot(component_range, bic_scores, label='BIC', marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.title('AIC and BIC Scores for Different Number of Components')
    plt.legend()
    plt.show()

    # Select the optimal number of components based on the lowest BIC
    optimal_n_components = component_range[np.argmin(bic_scores)]
    print(f'Optimal number of components: {optimal_n_components}')



def impute_knn(data_series, n_neighbors=range(1, 10), cv=5, figsize=(8,6), flag=True):
    """
    Impute missing values using KNN and find the optimal number of neighbors and weights.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the features to impute.
    features_list (list): List of feature names to impute.
    n_neighbors (range): Range of n_neighbors to try. Default is range(1, 10).
    cv (int): Number of cross-validation folds. Default is 5.
    figsize (tuple): Figure size for the plot. Default is (8, 6).

    Returns:
    pd.DataFrame: DataFrame with imputed values for the specified features.
    """

    # apply Min-Max Scaling
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_series), columns=data_series.columns)
    
    # Define the parameter grid for n_neighbors and weights
    if flag:
        param_grid = {'n_neighbors': n_neighbors, 'weights': ['uniform', 'distance']}
    else:
        param_grid = {'n_neighbors': n_neighbors, 'weights': ['uniform']}

    # Use GridSearchCV to find the best n_neighbors and weights
    grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=cv)
    grid_search.fit(data_scaled.dropna(), data_scaled.dropna())

    # Initialize variable & print
    best_params = grid_search.best_params_
    best_n_neighbors = best_params['n_neighbors']
    best_weights = best_params['weights']
    print(f"The best n_neighbors for KNN imputer is: {best_n_neighbors}")
    print(f"The best weights for KNN imputer is: {best_weights}")

    # Plot the results of GridSearchCV
    results = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=figsize)
    if flag:
        for weight in ['uniform', 'distance']:
            subset = results[results['param_weights'] == weight]
            plt.plot(n_neighbors, subset['mean_test_score'], label=f'weights={weight}')
    else:
        for weight in ['uniform']:
            subset = results[results['param_weights'] == weight]
            plt.plot(n_neighbors, subset['mean_test_score'], label=f'weights={weight}')        
    
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Mean Test Score')
    plt.title('Grid Search Results for KNN Imputer')
    plt.legend()
    plt.show()

    # Impute using the best n_neighbors and weights
    imputer = KNNImputer(n_neighbors=best_n_neighbors, weights=best_weights)
    imputed_data = imputer.fit_transform(data_scaled)

    # reverse scaling to original range
    imputed_data = pd.DataFrame(scaler.inverse_transform(imputed_data), columns=data_scaled.columns)
    
    # Return imputed DataFrame  
    return imputed_data