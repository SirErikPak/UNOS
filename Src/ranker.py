import pandas as pd
import NominalSurvivalRanker


def mean_survival_by_category(data, category_col, duration_col='TransplantSurvivalDay'):
    """
    Returns a DataFrame with each category, its mean survival days, and record counts.
    Warning-free and optimized for categorical columns.
    """
    out = (
        data[[category_col, duration_col]]
        .dropna(subset=[duration_col])
        .groupby(category_col, observed=True)[duration_col]
        .agg(['mean', 'median', 'max', 'min', 'count'])
        .reset_index()
        .rename(columns={
            "mean": "mean_survival_days",
            "median": "median_survival_days",
            "max": "maximum_survival_days",
            "min": "minimum_survival_days",
            "count": "record_count"
        })
    )

    out = out.sort_values(by="record_count", ascending=False)

    # print for display
    print(out.to_string(index=False))

    return

def cat_feature_ranker(
    data,
    type_value,       
    gender_value, 
    mapping_data,
    feature,
    class_values,
    re_run=False
):
    """
    Rank and collapse a categorical feature, update mapping_data,
    and attach type/gender metadata to the mapping row.
    """

    feature_name = feature[0]

    # --- housekeeping on re-run ---
    if re_run:
        if not mapping_data.empty and "feature" in mapping_data.columns:
            mapping_data = mapping_data[mapping_data["feature"] != feature_name]
        class_values['plot'] = False

    # --- run ranker ---
    ranker = NominalSurvivalRanker.NominalSurvivalRanker(**class_values)
    results = ranker.fit(data, feature_name)

    # --- build mapping row ---
    new_row = ranker.build_feature_dict(results)

    # inject type + gender metadata
    new_row["type"] = type_value
    new_row["gender"] = gender_value
    new_row["consolidate"] = False  # default to False; can be updated later based on analysis`

    # convert to DataFrame
    new_row = pd.DataFrame([new_row])

    # append to mapping_data
    mapping_data = pd.concat([mapping_data, new_row], ignore_index=True)

    # remove duplicates in mapping_data based on 'feature', 'type', and 'gender' columns,
    # keeping the last occurrence (most recent)
    if "feature" in mapping_data.columns:
        mapping_data = mapping_data.drop_duplicates(subset=["feature", "type", "gender"], keep="last").reset_index(drop=True)

    # --- display ---
    print_sorted_groups(results["mapping"])

    if not re_run:
        mean_survival_by_category(data, feature_name)

    return mapping_data, results
     

def print_sorted_groups(mapping):
    # mapping is a dict
    if isinstance(mapping, dict):
        data = pd.DataFrame(
            [{"category": k, "group": v} for k, v in mapping.items()]
        )

    #  mapping is a DataFrame from build_feature_dict
    elif isinstance(mapping, pd.DataFrame):
        # detect group columns dynamically
        group_cols = [c for c in mapping.columns if c.startswith("Group_") or c == "Other"]
        id_cols = [c for c in mapping.columns if c not in group_cols]

        # melt wide → long
        data = mapping.melt(
            id_vars=id_cols,
            value_vars=group_cols,
            var_name="group",
            value_name="category"
        ).dropna()

    else:
        raise ValueError("mapping must be a dict or DataFrame")

    # dynamic group detection and sorting
    real_groups = sorted(
        [g for g in data["group"].unique() if g.startswith("Group_")],
        key=lambda x: int(x.split("_")[1])
    )
    other_groups = [g for g in data["group"].unique() if not g.startswith("Group_")]

    # print header for display
    print(f"{':' * 30} GROUPED CATEGORIES {':' * 30}")

    # print real groups
    for g in real_groups:
        print(f"\n>>> {g} ({(data['group'] == g).sum()} categories)")
        cats = data[data["group"] == g]["category"].sort_values().tolist()
        for c in cats:
            print(f"  - {c}")

    # print Other last
    for g in other_groups:
        print(f"\n>>> {g} ({(data['group'] == g).sum()} categories)")
        cats = data[data["group"] == g]["category"].sort_values().tolist()
        for c in cats:
            print(f"  - {c}")
    print()


def update_remove_cols_and_mapping(mapping_data, remove_cols, feature, type_value, gender_value):
    """
    Update the remove_cols list with new feature columns (deduped) and
    remove the feature entry from mapping_data if present.
    """

    # Validate input
    if not feature:
        return mapping_data, (remove_cols or [])

    primary_feature = feature[0]

    # Remove feature row from mapping_data if present
    if (
        isinstance(mapping_data, pd.DataFrame)
        and not mapping_data.empty
        and "feature" in mapping_data.columns
    ):
        mask = mapping_data["feature"] == primary_feature
        if mask.any():
            mapping_data = mapping_data.loc[~mask].copy()

    # Ensure remove_cols exists
    if not isinstance(remove_cols, list):
        remove_cols = []

    # Append new feature columns
    remove_cols.extend(feature)

    # remove duplicates mapping_data based on 'feature', 'type', and 'gender' columns,
    # keeping the last occurrence (most recent)
    if isinstance(mapping_data, pd.DataFrame) and "feature" in mapping_data.columns:
        mapping_data = mapping_data.drop_duplicates(subset=["feature", "type", "gender"], keep="last").reset_index(drop=True)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for col in remove_cols:
        if col not in seen:
            seen.add(col)
            deduped.append(col)

    return mapping_data, deduped