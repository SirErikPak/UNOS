
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from collections import defaultdict
from scipy import stats


def any_nans(data: pd.DataFrame, txt: str='') -> None:
    """
    Identifies and displays columns with null values, along with their counts and percentages.
    Optimized for large datasets (e.g., PyArrow-backed) by using vectorized operations.
    
    Args:
        data: The pandas DataFrame to inspect.
    """
    total_rows = len(data)
    
    # 1. Get counts of nulls for all columns
    null_counts = data.isnull().sum()
    
    # 2. Filter for only columns that have at least one null
    null_counts = null_counts[null_counts > 0]
    
    if not null_counts.empty:
        print(f"--- Missing Values Found ({txt}) (Total Rows: {total_rows:,}) ---")
        
        # 3. Calculate percentage and build summary table
        percent = (null_counts / total_rows) * 100
        
        summary = pd.DataFrame({
            'Count': null_counts,
            'Percentage': percent.map("{:.4f}%".format)
        }).sort_values(by='Count', ascending=False)
        
        print(summary)
    else:
        print(f"Clean Dataset: No NaNs found across {total_rows:,} rows.")



def check_informative_missingness(data, col, txt='', target='TransplantSurvivalDay', unknown_val=None):
    """
    Compare survival outcomes between Known vs Missing/Unknown groups for one or more columns.
    Computes Welch's t-test, Cohen's d, and 95% CI for effect size.
    """

    # If col is a list, iterate cleanly
    if isinstance(col, (list, tuple)):
        for c in col:
            check_informative_missingness(data, c, txt=txt, target=target, unknown_val=unknown_val)
        return

    # --- Single-column logic ---
    # Missingness definition
    if unknown_val is not None:
        is_unknown = (data[col] == unknown_val) | (data[col].isna())
    else:
        is_unknown = data[col].isna()

    unknown = data.loc[is_unknown, target].dropna()
    known   = data.loc[~is_unknown, target].dropna()

    # Check sample size
    if len(unknown) < 2 or len(known) < 2:
        print(f"--- {col} ---")
        print("Insufficient data for T-test.\n")
        return

    # Group stats
    n_u, n_k = len(unknown), len(known)
    m_u, m_k = unknown.mean(), known.mean()
    var_u, var_k = unknown.var(ddof=1), known.var(ddof=1)

    # --- Cohen's d ---
    dof = n_u + n_k - 2
    pooled_std = np.sqrt(((n_u - 1) * var_u + (n_k - 1) * var_k) / dof)
    d = (m_u - m_k) / pooled_std if pooled_std != 0 else 0

    # 95% CI for d
    se_d = np.sqrt((n_u + n_k) / (n_u * n_k) + (d**2) / (2 * (n_u + n_k)))
    z = stats.norm.ppf(0.975)
    lower, upper = d - z * se_d, d + z * se_d

    # Effect size interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        strength = "Negligible"
    elif abs_d < 0.5:
        strength = "Small"
    elif abs_d < 0.8:
        strength = "Medium"
    else:
        strength = "Large"

    # --- Welch's t-test ---
    t_stat, p_val = stats.ttest_ind(unknown, known, equal_var=False)

    # --- Output ---
    print(f"--- Missingness ({txt}): {col} ---")
    print(f"Group Sizes:    (Unknown={n_u:,}, Known={n_k:,})")
    print(f"Mean survival:  (Unknown={m_u:,.1f}d, Known={m_k:,.1f}d)")
    print(f"Difference:     {m_u - m_k:,.1f} days\n")

    print("--- Welch's t-test Analysis ---")
    print(f"ρ-Value: {p_val:.4f}")

    if p_val < 0.05:
        print("RESULT: Missingness is associated with survival (informative missingness).")
        print("Interpretation: Not MCAR. Likely MAR or MNAR.")
    else:
        print("RESULT: No strong evidence that missingness affects survival.")
        print("Interpretation: Consistent with MCAR (but not definitive).")
    print()

    print("--- Cohen's Analysis ---")
    print(f"Cohen's d:     {d:.4f} ({strength})")
    print(f"95% CI:        [{lower:.4f}, {upper:.4f}]")

    if lower > 0 or upper < 0:
        print("Result: INFORMATIVE MISSINGNESS (statistically significant difference; small effect size)")
    else:
        print("Result: Likely Random Missingness (Effect size not significant)")
    print()


def continuous_value_predicts_survival(data, col, txt='', target='TransplantSurvivalDay'):
    """
    Evaluates the predictive power of a continuous feature against survival time 
    using linear (Pearson) and monotonic (Spearman) correlation metrics.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing both the feature and the survival target.
    col : str
        The name of the continuous feature column to evaluate.
    target : str, default='TransplantSurvivalDay'
        The continuous target variable representing survival duration.

    Returns:
    --------
    None
        Prints a summary of correlation coefficients, p-values, and 
        the approximate variance explained (R-squared).
    """
    # Remove rows with missing values in either the feature or the target
    known = data[[col, target]].dropna()

    # Pearson Correlation: Measures the linear relationship.
    # Assumes the data follows a normal distribution and the relationship is a straight line.
    pearson_r, pearson_p = stats.pearsonr(known[col], known[target])
    
    # Spearman Correlation: Measures the monotonic relationship.
    # Uses the rank of the data points; robust to outliers and non-linear (but directional) curves.
    spearman_r, spearman_p = stats.spearmanr(known[col], known[target])

    print(f"--- Value Predicts Survival ({txt}): {col} ---")
    
    # Pearson R^2 (Coefficient of Determination)
    # Represents the proportion of variance in survival explained by the linear model of the feature.
    print(f"Pearson r={pearson_r:.3f}, p={pearson_p:.4g} & Approx. Variance Explained={pearson_r**2:.3%}")
    
    # Spearman R^2 
    # Represents the proportion of variance in the *ranks* of survival explained by the feature ranks.
    print(f"Spearman r={spearman_r:.3f}, p={spearman_p:.4g} & Approx. Variance Explained={spearman_r**2:.3%}")
    print()


def get_feature_info(data, colstr, cat=False):
    # find matching columns
    features = sorted(data.columns[data.columns.str.contains(colstr)].tolist())

    # describe block
    print(data[features].describe(include='all').T.to_string())
    print("\n:::: NaN Count:")
    print(data[features].isna().sum().sort_index().to_string(), "\n")

    if cat:
        for col in features:
            print(f"--- {col} ---")
            print("dtype:", data[col].dtype)

            # pandas categorical
            if str(data[col].dtype).startswith("category"):
                print("Categories:", list(data[col].cat.categories))
                print("Ordered:", data[col].cat.ordered)
            else:
                print("Not categorical or dictionary-encoded.")

            print()

    return features


def get_top_frequencies(data, column_name, top_n=20, sep=","):
    """
    Explodes a string-delimited column into individual items and calculates 
    their frequency distribution.

    This is particularly useful for multi-label data (like medications or 
    crime tags) where a single record may contain multiple categories.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataframe containing the data.
    column_name : str
        The name of the column to process (must contain strings or NaNs).
    top_n : int, default 20
        The number of most frequent items to return.
    sep : str, default ","
        The delimiter used to separate items in the string.

    Returns:
    --------
    list of tuples
        A list of (item, count) pairs for the top_n most frequent items.
    """
    
    # 1. Handle missing values and split into lists
    # 2. Explode the lists into individual rows
    # 3. Strip whitespace to ensure 'Meds' and ' Meds' are counted together
    all_items = (
        data[column_name]
        .dropna()
        .str.split(sep)
        .explode()
        .str.strip()
    )

    # Calculate frequencies using Counter
    freq = Counter(all_items)

    return freq.most_common(top_n)


def build_feature_dict(results):
    """
    Construct a clean, deterministic summary of collapsed categorical groups.

    Parameters
    ----------
    results : dict
        Expected keys:
        - "feature": str
        - "mapping": dict {raw_category -> group_label}
        - "counts": dict {raw_category -> count}
        - "method_used": str

    Returns
    -------
    dict
        {
            "feature": <feature_name>,
            "method_used": <method>,
            "Group_1": {"Total_N": int, "Categories": [...]},
            ...
        }
    """

    feature = results["feature"]
    mapping = results["mapping"]
    counts = results["counts"]
    method_used = results.get("method_used", "unknown")

    # Reverse mapping: group → list of categories
    rev = defaultdict(list)
    for cat, grp in mapping.items():
        rev[grp].append(cat)

    # Sort categories inside each group for deterministic output
    for grp in rev:
        rev[grp] = sorted(rev[grp])

    # Compute group totals
    group_totals = {
        grp: sum(int(counts.get(cat, 0)) for cat in cats)
        for grp, cats in rev.items()
    }

    # Build group summaries
    groups = {
        grp: {
            "Total_N": int(group_totals[grp]),
            "Categories": cats
        }
        for grp, cats in rev.items()
    }

    # Sort groups by numeric suffix (Group_1, Group_2, ...)
    sorted_groups = dict(sorted(
        groups.items(),
        key=lambda x: int(x[0].split("_")[1])
    ))

    # Final deterministic structure
    return {
        "feature": feature,
        "method_used": method_used,
        **sorted_groups
    }


def write_to_file(data, filename, path='../Data/', format='csv'):
    """
    write dataframe to disk
    """
    # intialize variable
    file_path = path + filename + f".{format}"

    if format.lower() == 'csv':
        # write to disk
        data.to_csv(file_path, index=False)
    else:
        data.to_pickle(file_path)     
    
    return print(f"{len(data):,} records written to {file_path}")