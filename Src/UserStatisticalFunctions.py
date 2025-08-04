# import library
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency
from itertools import combinations
from scipy.stats import mannwhitneyu
from scipy.stats import chisquare
from scipy.stats import ks_2samp
from pyampute.exploration.mcar_statistical_tests import MCARTest


def chi_square_test(data_prior, data_after):
    """
    Perform a Chi-Square Goodness-of-Fit Test to evaluate whether imputation 
    has significantly altered the distribution of data. 

    None
        Prints the Chi-Square statistic, p-value, and an interpretation of 
        whether the imputation has significantly changed the distribution.
        (Parametric & same shape required)
    """
    # Observed (imputed data) vs. Expected (original data) frequencies
    observed = data_after.dropna().values
    expected = data_prior.dropna().values
    
    # Perform Chi-Square Goodness-of-Fit Test
    chi_stat, p_value = chisquare(observed, expected)
    
    # display results
    print(f"Chi-Square Statistic: {chi_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # print Interpretation
    if p_value > 0.05:
        print("✅ Imputation preserved the original distribution.")
    else:
        print("⚠️ Imputation significantly altered the original distribution.")



def imputation_test_KS(data_prior, data_after):
    """
    Perform the Kolmogorov-Smirnov (KS) test to compare the distributions of original and imputed data.

    This function evaluates whether the imputation process has significantly altered the distribution 
    of each feature by comparing the non-missing values from the original dataset with the imputed values.

    Notes:
    ------
    - The KS test is a non-parametric test that compares two distributions.
    - A low p-value (≤ 0.05) suggests that the distribution of imputed values differs significantly 
      from the original non-missing values.
    - The function assumes that `features` is a predefined list of column names to test.
    - Non-Parametric
    """
    # get features
    features = data_prior.columns
    
    # perform KS test for each imputed feature
    for feature in features:
        original_values = data_prior.dropna()  # Non-missing values
        imputed_values = data_after  # Imputed values
        # test
        ks_stat, p_value = ks_2samp(original_values, imputed_values)
        # display
        print(f"Feature: {feature} | KS Statistic: {ks_stat:.4f} | p-value: {p_value:.4f}")
        
        # print Interpretation
        if p_value > 0.05:
            print("✅ Imputation preserved the distribution.")
        else:
            print("⚠️ Imputation may have altered the distribution.")



def correlation_with_threshold(df, method='pearson', threshold=0.9, flag=False):
    """
    This function is designed to identify pairs of features that are highly correlated in a dataset. 
    It calculates the correlation matrix of numerical columns and identifies pairs of features where 
    the absolute correlation is greater than a given threshold (default is 0.9).
    """
    # initilaize variable
    feature = list()
    # calculate the correlation matrix
    correlation_matrix = df.select_dtypes(exclude='object').corr(method=method)
    
    # get the number of features
    num_features = correlation_matrix.shape[0]
    
    # iterate over the upper triangular part of the matrix
    for i in range(num_features):
        for j in range(i+1, num_features):
            feature1 = correlation_matrix.index[i]
            feature2 = correlation_matrix.columns[j]
            correlation = correlation_matrix.iloc[i, j]
            if abs(correlation) > threshold:
                feature.append(feature2)
                print(f"Correlation between {feature1} and {feature2}: {correlation:.3f}")

    if flag:
        return feature
    else:
        return

    
def mannwhitneyu_combinations(data, features):
    """
    The function performs the Mann-Whitney U test on all combinations of two features 
    in a given list of features. The test checks if there is a significant difference 
    between the distributions of two independent variables. It prints and returns the
    U-statistic and p-value for each pair of features. The null hypothesis for the 
    Mann-Whitney U test is that the two populations being compared have the same distribution. 
    It assumes that there is no difference between the two groups in terms of their central tendency 
    or distribution. The alternative hypothesis is that the distributions of the two populations
    are not identical.
    """
    # ititaize DataFrame
    resultDF = pd.DataFrame(columns=['column1', 'column2', 'U_statistic', 'p_value'])
    
    # iterate over all combinations of features
    for feature1, feature2 in combinations(features, 2):
        # extract the data for the two features
        group1 = data[feature1].dropna()  # remove NaN values
        group2 = data[feature2].dropna()  # remove NaN values
        
        # perform the Mann-Whitney U test
        U_statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        # store the result
        rowValue = {'column1': feature1, 'column2': feature2, 'U_statistic': U_statistic, 'p_value': p_value}
        resultDF.loc[len(resultDF)] = rowValue

    # sort DataFrame
    resultDF  = resultDF.sort_values(by='p_value', ascending=False)

    return resultDF



def spearman_rank_correlation(data, col1, col2):
    """
    Spearman's rank correlation is a non-parametric measure of the monotonicity of the relationship between two variables.
    """
    data = data.dropna()
    correlation, p_value = stats.spearmanr(data[col1], data[col2])
    r, p = stats.pearsonr(data[col1], data[col2])

    print(f"Spearman correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}\n")
    print(f"Pearson correlation coefficient: {r:.4f}")
    print(f"P-value: {p:.4f}")



def multi_independence_category(data, cat_col):
    """
    Hypothesis testing using Chi-square statistic and calculating Cramer's V to 
    build consensus for all the categorical variables. 
    """
    # initialize variables
    results = []
    valid_cols = [col for col in cat_col if col in data.columns]

    # use combinations to get unique pairs of columns
    for col1, col2 in combinations(valid_cols, 2):
        # create a contingency table
        contingency_table = pd.crosstab(data[col1], data[col2])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        # total number of observations
        n = contingency_table.values.sum()
        # get the number of categories in each variable (rows and columns)
        r, k = contingency_table.shape
        min_dim = min(r-1, k-1)
        
        # handle division by zero
        if n * min_dim == 0:
            cramer_v = np.nan
        else:
            cramer_v = np.sqrt(chi2 / (n * min_dim))
        
        results.append({
            'column1': col1,
            'column2': col2,
            'chi2': chi2,
            'p_value': p_value,
            'cramer_v': cramer_v
        })
    
    return pd.DataFrame(results).sort_values(by='cramer_v', ascending=False)



def independence_test_category(data, cat1, cat2, flag=False):
    """
    Hypothesis testing using Ch-square statistic and calculating Cramer's V to 
    build consensus for two categorical variables.
    """
    # create a contingency table
    contingency_table = pd.crosstab(data[cat1], data[cat2])
    # perform Chi-square test
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    
    # total number of observations
    n = contingency_table.sum().sum()
    # get the number of categories in each variable (rows and columns)
    r, k = contingency_table.shape
    # calculate Cramer's V
    cramer_v = np.sqrt(chi2 / (n * min(k-1, r-1)))

    # display
    print(f"Test of Independence for Catergorical Variables: {cat1} & {cat2}")
    print(f"Chi-square statistic: {chi2:,.2f}")
    print(f"p-value: {p_value:,.4f}")
    print(f"Cramer's V: {cramer_v:,.4f}")

    if flag:
        return contingency_table



def test_mcar(data):
    """
    Perform Little's MCAR (Missing Completely At Random) test on a dataset.

    This function uses Little's MCAR test to determine if the missing data in the provided
    dataset is missing completely at random. It prints the p-value of the test and interprets
    the result.

    Parameters:
    data (pd.DataFrame): The dataset to test for MCAR. It should be a pandas DataFrame.

    Returns:
    None

    Raises:
    ValueError: If the input data is not a pandas DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    try:
        # perform Little's MCAR test
        mcar_test = MCARTest(method="little")
        p_value = mcar_test.little_mcar_test(data)
        print(f"Little's MCAR test p-value: {p_value:.4f}")
        
        # interpretation
        if p_value > 0.05:
            print("✅ Fail to reject null hypothesis: Data may be Missing Completely At Random (MCAR).")
        else:
            print("⚠️ Reject null hypothesis: Data is not Missing Completely At Random (MCAR).")
    except Exception as e:
        print(f"An error occurred: {e}")
