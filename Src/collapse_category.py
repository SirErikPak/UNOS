from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def nominal_rank_survival(data, feature_col, duration_col, custom_order=None, min_n=30,
                           r_thresh=0.10, rotation=0):
    """
    Perform survival-based ranking and statistical grouping of categorical features.

    This function evaluates the relationship between a categorical feature and a 
    continuous outcome (e.g., survival time) using nonparametric statistical methods.
    It identifies statistically distinct categories and automatically collapses 
    those that are survival-equivalent based on hypothesis testing and effect size.

    Core functionality:
    -------------------
    1. Filters categories based on minimum sample size (min_n)
    2. Performs global comparison using Kruskal–Wallis test
    3. Conducts pairwise Mann–Whitney U tests with Bonferroni correction
    4. Computes effect sizes (rank-biserial correlation)
    5. Collapses statistically similar categories using graph-based clustering
    6. Returns a structured summary for downstream analysis

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing feature and outcome columns

    feature_col : str
        Categorical feature to evaluate

    duration_col : str
        Continuous outcome variable (e.g., survival time)

    custom_order : list, optional
        User-defined ordering of categories; defaults to sorted order

    min_n : int, default=30
        Minimum number of observations required per category

    r_thresh : float, default=0.10
        Effect size threshold for collapsing categories.
        Categories are merged if:
            - Pairwise difference is NOT statistically significant AND
            - Effect size (|r|) is below this threshold (negligible)

    Returns
    -------
    dict or None
        Dictionary containing:
        - feature: analyzed feature name
        - groups: collapsed category groups
        - mapping: category → group mapping
        - pairwise: pairwise statistical results
        - kruskal: global test statistics (H, p, eta²)
        - counts: original category counts

        Returns None if no categories meet the minimum sample size.

    Notes
    -----
    - Kruskal–Wallis is used for nonparametric group comparison
    - Mann–Whitney U is used for pairwise testing
    - Graph-based clustering ensures transitive merging of equivalent categories
    - Designed for survival analysis, feature engineering, and statistical reporting
    """

    # Remove rows with missing outcome values and ensure categorical feature is string
    temp_df = data.dropna(subset=[duration_col]).copy()
    temp_df[feature_col] = temp_df[feature_col].astype(str)

    print(f"TOTAL RECORDS (raw): {len(data):,}")
    print(f"TOTAL RECORDS (non-missing {duration_col}): {len(temp_df):,}")

    # Retain only categories with sufficient sample size
    counts = temp_df[feature_col].value_counts()
    valid = counts[counts >= min_n].index.tolist()
    filtered = temp_df[temp_df[feature_col].isin(valid)].copy()

    print(f"TOTAL INCLUDED RECORDS (n ≥ {min_n}): {len(filtered):,}")

    if filtered.empty:
        print(f"SKIPPED: No categories in {feature_col} have n ≥ {min_n}")
        return None

    # Determine Custom Category Order
    present = filtered[feature_col].unique()
    if isinstance(custom_order, list):
        order = [c for c in custom_order if c in present]
    else:
        order = sorted(present)

    # Create a dictionary mapping each category to its outcome values
    group_map = {
        cat: grp[duration_col].values
        for cat, grp in filtered.groupby(feature_col)
    }

    groups = [group_map[c] for c in order]

    # KRUSKAL–WALLIS
    # Tests whether at least one group differs significantly
    H, p = kruskal(*groups)
    k = len(groups)
    n = len(filtered)

    # Effect size (eta-squared approximation for Kruskal)
    eta2 = (H - k + 1) / (n - k)

    # Violin plot to show distribution differences across categories
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=filtered,
        x=feature_col,
        y=duration_col,
        order=order,
        hue=feature_col,
        palette="viridis",
        inner="quartile",
        legend=False
    )
    plt.title(
        f"Survival Relevance (n ≥ {min_n}): {feature_col}\n"
        f"H={H:.2f}, $\\eta^2$={eta2:.3f}"
    )
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()

    # Reporting
    print("=" * 65)
    print(f"RANKING REPORT: {feature_col} (n ≥ {min_n})")
    print("=" * 65)

    print("INCLUDED CATEGORIES:")
    for c in order:
        print(f" - {c:<15}: {counts[c]:>6} records")

    # Identify excluded (low-sample) categories
    bypassed = counts[counts < min_n]
    if not bypassed.empty:
        print(f"(Bypassed {len(bypassed):,} categories with n < {min_n})")

    print("-" * 65)
    print(f"Kruskal H: {H:.4f}")
    print(f"P-value:   {p:.6e}")
    print(f"Effect η²: {eta2:.4f}")
    print("-" * 65)

    # PAIRWISE TESTING
    # Only performed if the global test is significant
    pairwise_results = []

    if p < 0.05:
        pairs = list(combinations(order, 2))
        alpha = 0.05 / len(pairs)  # Bonferroni correction

        print(f"Pairwise Testing (Bonferroni Alpha: {alpha:.4f})")

        for c1, c2 in pairs:
            g1, g2 = group_map[c1], group_map[c2]

            # Mann–Whitney U test
            stat, p_pair = mannwhitneyu(g1, g2)

            # Rank-biserial correlation (effect size)
            r = 1 - (2 * stat) / (len(g1) * len(g2))

            sig = p_pair < alpha

            print(
                f"{c1:<12} vs {c2:<12} | "
                f"p={p_pair:.4e} | "
                f"r={r:>6.3f} | "
                f"{'*Sig*' if sig else 'Not Sig'}"
            )

            pairwise_results.append({
                "cat1": c1,
                "cat2": c2,
                "p": p_pair,
                "r": r,
                "sig": sig
            })

    print("=" * 65 + "\n")

    # GRAPH-BASED CATEGORY COLLAPSING
    # Build a graph where edges connect statistically equivalent categories
    G = nx.Graph()
    G.add_nodes_from(order)

    for row in pairwise_results:
        if (not row["sig"]) and abs(row["r"]) < r_thresh:
            # Merge only if NOT significant AND effect size is negligible
            G.add_edge(row["cat1"], row["cat2"])

    # Connected components = collapsed groups
    collapsed_groups = [
        sorted(list(comp)) for comp in nx.connected_components(G)
    ]

    # HANDLE SMALL CATEGORIES
    # Add excluded categories as a separate group
    small_cats = sorted(bypassed.index.tolist())
    if small_cats:
        collapsed_groups.append(small_cats)

    # Map each original category to its new grouped label
    collapse_map = {}
    for i, group in enumerate(collapsed_groups):
        for cat in group:
            collapse_map[cat] = f"Group_{i+1}"

    # RETURN STRUCTURED RESULT
    return {
        "feature": feature_col,
        "groups": collapsed_groups,
        "mapping": collapse_map,
        "pairwise": pairwise_results,
        "kruskal": {"H": H, "p": p, "eta2": eta2},
        "counts": counts.to_dict()
    }


def results_to_single_dataframe(results):
    """
    Convert a structured statistical analysis result into a standardized long-format DataFrame.

    This function flattens a nested result dictionary (e.g., from clustering or 
    hypothesis testing output) into a single pandas DataFrame with three columns:
        - section: high-level category (feature, groups, mapping, etc.)
        - key: specific identifier within the section
        - value: corresponding value (scalar or dictionary)

    Parameters
    ----------
    results : dict
        Dictionary containing analysis results with the following expected structure:
        {
            "feature": str,
            "groups": list,
            "mapping": dict,
            "pairwise": list of dicts,
            "kruskal": dict,
            "counts": dict
        }

    Returns
    -------
    pandas.DataFrame
        A long-format DataFrame where each row represents a single piece of information.
        This format is useful for:
        - logging results
        - exporting to CSV/JSON
        - downstream aggregation or visualization

    Notes
    -----
    - Pairwise comparisons are stored as dictionaries inside the 'value' column.
    - Numeric values are explicitly cast to ensure consistency.
    - This structure is especially useful for statistical reporting pipelines.
    """
    # initialize list
    rows = []

    # Store the primary feature being analyzed
    rows.append({
        "section": "feature",
        "key": "feature",
        "value": results["feature"]
    })

    # Store group labels (e.g., category groupings)
    for i, group in enumerate(results["groups"], start=1):
        rows.append({
            "section": "groups",
            "key": f"Group_{i}",
            "value": group
        })

    # Maps original categories to grouped labels
    for cat, grp in results["mapping"].items():
        rows.append({
            "section": "mapping",
            "key": cat,
            "value": grp
        })

    # Stores statistical test results between category pairs
    for row in results["pairwise"]:
        rows.append({
            "section": "pairwise",
            "key": f"{row['cat1']} vs {row['cat2']}",
            "value": {
                "p": float(row["p"]),      # p-value
                "r": float(row["r"]),      # effect size (e.g., rank-biserial)
                "sig": bool(row["sig"])    # significance flag
            }
        })

    # Global test statistics across all groups
    for k, v in results["kruskal"].items():
        rows.append({
            "section": "kruskal",
            "key": k,
            "value": float(v)
        })

    # Number of observations per category/group
    for cat, n in results["counts"].items():
        rows.append({
            "section": "counts",
            "key": cat,
            "value": int(n)
        })

    # Convert accumulated rows into a pandas DataFrame
    return pd.DataFrame(rows)