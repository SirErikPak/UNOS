
import pandas as pd
from scipy import stats

def any_nans(data: pd.DataFrame) -> None:
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
        print(f"--- Missing Values Found (Total Rows: {total_rows:,}) ---")
        
        # 3. Calculate percentage and build summary table
        percent = (null_counts / total_rows) * 100
        
        summary = pd.DataFrame({
            'Count': null_counts,
            'Percentage': percent.map("{:.4f}%".format)
        }).sort_values(by='Count', ascending=False)
        
        print(summary)
    else:
        print(f"Clean Dataset: No NaNs found across {total_rows:,} rows.")



def check_informative_missingness(data, col, target='TransplantSurvivalDay', unknown_val=None):
    """
    Compares survival days between 'Known' and 'Unknown' groups.
    Accepts either a single column name or a list of column names.
    """

    # If col is a list, loop through each column
    if isinstance(col, (list, tuple)):
        for c in col:
            check_informative_missingness(data, c, target=target, unknown_val=unknown_val)
        return  # prevent running the rest of the function on the list itself

    # --- Single column logic below ---
    # Define unknown mask
    if unknown_val is not None:
        is_unknown = (data[col] == unknown_val) | (data[col].isna())
    else:
        is_unknown = data[col].isna()

    # Extract survival values
    unknown = data.loc[is_unknown, target].dropna()
    known   = data.loc[~is_unknown, target].dropna()

    # Not enough data for a t-test
    if len(unknown) < 2 or len(known) < 2:
        print(f"--- {col} ---")
        print("Insufficient data for T-test.\n")
        return

    # Welch's t-test
    t_stat, p_val = stats.ttest_ind(unknown, known, equal_var=False)

    # Display results
    print(f"--- {col} ---")
    print(f"n: (Unknown={len(unknown):,}, Known={len(known):,})")
    print(f"Mean survival: (Unknown={unknown.mean():,.1f}d, Known={known.mean():,.1f}d)")
    print(f"Difference: {unknown.mean() - known.mean():,.1f} days")
    print(f"ρ-Value: {p_val:.4f}")

    if p_val < 0.05:
        print("RESULT: Statistically Significant (Informative Missingness) consistent with MAR or MNAR")
    else:
        print("RESULT: Not Significant (Likely Random Missingness) consistent with MCAR")
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

    return