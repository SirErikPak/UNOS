import pandas as pd
import NominalSurvivalRanker


import pandas as pd
import NominalSurvivalRanker

# Elegant presentation layer engines
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.box import ROUNDED
from rich.text import Text


def mean_survival_by_category(data: pd.DataFrame, category_col: str, duration_col: str = 'TransplantSurvivalDay') -> pd.DataFrame:
    """
    Returns a DataFrame with each category, its mean survival days, and record counts.
    Warning-free, optimized for categorical columns, and printed with modern Rich styling.
    """
    console = Console()
    
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

    # --- Modern Visual Table ---
    table = Table(
        box=ROUNDED,
        border_style="dim",
        header_style="bold magenta",
        expand=False
    )
    
    table.add_column(category_col, justify="left", style="bold white")
    table.add_column("Mean Days", justify="right", style="cyan")
    table.add_column("Median Days", justify="right", style="green")
    table.add_column("Min Days", justify="right", style="dim white")
    table.add_column("Max Days", justify="right", style="dim white")
    table.add_column("Record Count", justify="right", style="yellow")
    
    for _, row in out.iterrows():
        table.add_row(
            str(row[category_col]),
            f"{row['mean_survival_days']:.1f}",
            f"{row['median_survival_days']:.1f}",
            f"{row['minimum_survival_days']:.1f}",
            f"{row['maximum_survival_days']:.1f}",
            f"{int(row['record_count'])}"
        )
        
    console.print(table)
    return out


def cat_feature_ranker(
    data: pd.DataFrame,
    type_value: str,       
    gender_value: str, 
    mapping_data: pd.DataFrame,
    feature: list,
    class_values: dict,
    re_run: bool = False
):
    """
    Rank and collapse a categorical feature, update mapping_data,
    and attach type/gender metadata to the mapping row with modern outputs.
    """
    feature_name = feature[0]
    local_class_values = class_values.copy()

    # --- housekeeping on re-run ---
    if re_run:
        if not mapping_data.empty and "feature" in mapping_data.columns:
            mapping_data = mapping_data[mapping_data["feature"] != feature_name]
        local_class_values['plot'] = False

    # Ensure our statistical power constraint bypass parameter defaults safely
    if "ignore_sig_on_low_effect" not in local_class_values:
        local_class_values["ignore_sig_on_low_effect"] = True

    # --- run ranker ---
    ranker = NominalSurvivalRanker.NominalSurvivalRanker(**local_class_values)
    results = ranker.fit(data, feature_name)

    if results is None:
        return mapping_data, None

    # --- build mapping row ---
    new_row = ranker.build_feature_dict(results)

    # inject type + gender metadata
    new_row["type"] = type_value
    new_row["gender"] = gender_value
    new_row["consolidate"] = False  # default to False; can be updated later based on analysis

    # convert to DataFrame
    new_row_df = pd.DataFrame([new_row])

    # append to mapping_data
    mapping_data = pd.concat([mapping_data, new_row_df], ignore_index=True)

    # remove duplicates in mapping_data based on 'feature', 'type', and 'gender' columns,
    # keeping the last occurrence (most recent)
    if "feature" in mapping_data.columns:
        mapping_data = mapping_data.drop_duplicates(subset=["feature", "type", "gender"], keep="last").reset_index(drop=True)

    # --- display ---
    print_sorted_groups(results["mapping"])

    if not re_run:
        mean_survival_by_category(data, feature_name, duration_col=ranker.duration_col)

    return mapping_data, results
     

def print_sorted_groups(mapping):
    """Prints an ultra-clean, structured overview of the collapsed group categories."""
    console = Console()
    
    # mapping is a dict
    if isinstance(mapping, dict):
        data = pd.DataFrame(
            [{"category": k, "group": v} for k, v in mapping.items()]
        )

    # mapping is a DataFrame from build_feature_dict
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

    table = Table(
        title="[bold white]:::::::::::::::::::::::::::::: GROUPED CATEGORIES ::::::::::::::::::::::::::::::[/bold white]",
        title_justify="center",
        box=None,
        show_header=False,
        padding=(0, 2)
    )

    # process and parse real groups
    for g in real_groups:
        cats = data[data["group"] == g]["category"].sort_values().tolist()
        cats_str = ", ".join([f"[yellow]'{c}'[/yellow]" for c in cats])
        grp_title = f"[bold cyan]▶ {g}[/bold cyan] [dim]({len(cats)} cats):[/dim]"
        table.add_row(grp_title, cats_str)

    # process and parse Other groups last
    for g in other_groups:
        cats = data[data["group"] == g]["category"].sort_values().tolist()
        cats_str = ", ".join([f"[dim white]'{c}'[/dim white]" for c in cats])
        grp_title = f"[bold white]▶ {g}[/bold white] [dim]({len(cats)} cats):[/dim]"
        table.add_row(grp_title, cats_str)

    console.print(table)
    console.print()


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
        mask = (mapping_data["feature"] == primary_feature) & \
               (mapping_data["type"] == type_value) & \
               (mapping_data["gender"] == gender_value)
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