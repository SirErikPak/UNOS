from sklearn.mixture import GaussianMixture
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def gaussian_mixture_components(data, column_list, seed, txt='', 
                                components_range=range(1, 11), figsize=(8,6), n_init=20):
    """
    Determines the optimal number of components for a Gaussian Mixture Model (GMM)
    using AIC and BIC for a single feature or multi-feature dataset.
    Uses AIC and BIC to evaluate model quality and prints:

    Note:
    The function fits GMMs with different numbers of components to the provided data,
    computes their AIC and BIC scores, and plots these scores to help visualize the optimal
    number of components. The optimal number is determined based on the lowest BIC score,
    as BIC tends to favor models that generalize well to unseen data by penalizing complexity
    more strongly than AIC.
    """

    # Auto-convert Series to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Subset and drop NaNs
    X = data[column_list].dropna()

    # Print feature info
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature list: {list(X.columns)}\n")

    # Convert to numpy
    X_arr = X.values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_arr)

    # Initialize AIC/BIC lists
    aic = []
    bic = []

    # Fit GMM for each number of components
    for n in components_range:
        gmm = GaussianMixture(
            n_components=n,
            n_init=n_init,     # run from different random initializations
            random_state=seed
        )
        gmm.fit(X_scaled)

        aic.append(gmm.aic(X_scaled))
        bic.append(gmm.bic(X_scaled))

    # Plot AIC/BIC
    plt.figure(figsize=figsize)
    plt.plot(components_range, aic, label="AIC", marker="o")
    plt.plot(components_range, bic, label="BIC", marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("AIC / BIC")
    plt.title(f"AIC and BIC for Gaussian Mixture Models {txt}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Best model = lowest BIC
    best_n = components_range[np.argmin(bic)]

    # Refit the best model
    best_gmm = GaussianMixture(
        n_components=best_n,
        n_init=n_init,
        random_state=seed
    )
    best_gmm.fit(X_scaled)
    # Posterior probabilities (also called the responsibility)
    posteriors = best_gmm.predict_proba(X_scaled)
    # Set global precision
    np.set_printoptions(precision=4, suppress=True)
    print(posteriors[:5])  # Show first 5 rows of posterior probabilities
    print(f"Optimal number of components (BIC): {best_n}")
    
    return best_n, posteriors





def autotune_n_init(
    X,
    n_components,
    grid=(1, 2, 5, 10, 20, 50),
    repeats=5,
    var_tol=1e-2,
    improv_tol=1e-1,
    base_seed=0
):
    """
    Auto-tune n_init by increasing it until:
      - log-likelihood variance across repeats is small, and
      - improvement in mean log-likelihood vs previous n_init is small.
    """
    def sample_ll(X, n_components, n_init, repeats, base_seed):
        ll = []
        for r in range(repeats):
            gmm = GaussianMixture(
                n_components=n_components,
                n_init=n_init,
                random_state=base_seed + r
            )
            gmm.fit(X)
            ll.append(gmm.score(X) * X.shape[0])  # total log-likelihood
        return np.array(ll)

    prev_mean = None
    history = []

    for n_init in grid:
        ll = sample_ll(X, n_components, n_init, repeats, base_seed)
        mean_ll = ll.mean()
        var_ll = ll.var()
        improv = np.inf if prev_mean is None else mean_ll - prev_mean

        history.append((n_init, mean_ll, var_ll, improv))

        if var_ll < var_tol and abs(improv) < improv_tol:
            return n_init, history

        prev_mean = mean_ll

    return grid[-1], history



from sklearn.metrics import silhouette_score

def gaussian_mixture_components_x(
    data, 
    column_list, 
    seed, 
    txt='', 
    components_range=range(2, 11), # Silhouette requires at least 2 clusters
    figsize=(10,6), 
    n_init_grid=(1, 2, 5, 10, 20, 50)
):
    # ... [Keep your existing preprocessing/scaling code here] ...

    # Lists for metrics
    aic, bic, sil = [], [], []
    tuned_n_init = {}

    for n in components_range:
        # Auto-tune n_init (Note: modified sample_ll to use .score() without multiplying by N)
        best_n_init, _ = autotune_n_init(
            X_scaled, n_components=n, grid=n_init_grid, base_seed=seed
        )
        tuned_n_init[n] = best_n_init

        gmm = GaussianMixture(n_components=n, n_init=best_n_init, random_state=seed)
        labels = gmm.fit_predict(X_scaled)

        aic.append(gmm.aic(X_scaled))
        bic.append(gmm.bic(X_scaled))
        # Silhouette score measures cluster cohesion vs separation
        sil.append(silhouette_score(X_scaled, labels))

    # Plotting
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(components_range, aic, label="AIC", marker="o", color='blue')
    ax1.plot(components_range, bic, label="BIC", marker="o", color='green')
    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("AIC / BIC")
    
    ax2 = ax1.twinx()
    ax2.plot(components_range, sil, label="Silhouette", marker="s", color='red', linestyle='--')
    ax2.set_ylabel("Silhouette Score (Higher is better)")
    
    plt.title(f"GMM Model Selection Metrics {txt}")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()

    best_n = components_range[np.argmin(bic)]
    
    return best_n, tuned_n_init[best_n], sil



def gaussian_mixture_components_x(
    data,
    column_list,
    seed,
    txt='',
    components_range=range(1, 11),
    figsize=(8,6),
    n_init_grid=(1, 2, 5, 10, 20, 50)
):
    """
    Determines the optimal number of components for a Gaussian Mixture Model (GMM)
    using AIC and BIC, with auto-tuned n_init based on log-likelihood stability.
    """

    # Auto-convert Series to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Subset and drop NaNs
    X = data[column_list].dropna()

    print(f"Number of features: {X.shape[1]}")
    print(f"Feature list: {list(X.columns)}\n")

    # Convert to numpy
    X_arr = X.values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_arr)

    # Initialize AIC/BIC lists
    aic = []
    bic = []
    tuned_n_init = {}

    # Fit GMM for each number of components
    for n in components_range:

        # Auto-tune n_init for this component count
        best_n_init, hist = autotune_n_init(
            X_scaled,
            n_components=n,
            grid=n_init_grid,
            repeats=5,
            var_tol=1e-2,
            improv_tol=1e-1,
            base_seed=seed
        )
        tuned_n_init[n] = best_n_init

        print(f"n_components={n}: auto-tuned n_init={best_n_init}")

        gmm = GaussianMixture(
            n_components=n,
            n_init=best_n_init,
            random_state=seed
        )
        gmm.fit(X_scaled)

        aic.append(gmm.aic(X_scaled))
        bic.append(gmm.bic(X_scaled))

    # Plot AIC/BIC
    plt.figure(figsize=figsize)
    plt.plot(components_range, aic, label="AIC", marker="o")
    plt.plot(components_range, bic, label="BIC", marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("AIC / BIC")
    plt.title(f"AIC and BIC for Gaussian Mixture Models {txt}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Best model = lowest BIC
    best_n = components_range[np.argmin(bic)]
    best_n_init = tuned_n_init[best_n]

    print(f"\nOptimal number of components (BIC): {best_n}")
    print(f"Auto-tuned n_init for best model: {best_n_init}\n")

    # Refit the best model with its tuned n_init
    best_gmm = GaussianMixture(
        n_components=best_n,
        n_init=best_n_init,
        random_state=seed
    )
    best_gmm.fit(X_scaled)

    # Posterior probabilities
    posteriors = best_gmm.predict_proba(X_scaled)
    labels = posteriors.argmax(axis=1)


    np.set_printoptions(precision=4, suppress=True)
    print(posteriors[:5])

    return best_n, best_n_init, posteriors