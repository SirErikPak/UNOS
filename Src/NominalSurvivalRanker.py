import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import combinations
from collections import defaultdict
from scipy.stats import kruskal, mannwhitneyu


class NominalSurvivalRanker:
    """
    Rank and collapse nominal categorical levels based on survival outcome similarity.

    This class is designed for exploratory feature engineering and survival-aware
    category reduction. It identifies categories that behave similarly with respect
    to a survival duration variable and optionally collapses them into broader groups.

    Statistical workflow
    --------------------
    1. Filters categories with sample size < min_n
    2. Performs omnibus Kruskal–Wallis test across retained categories
    3. Runs all pairwise Mann–Whitney U tests
    4. Applies multiple-testing correction:
         - Bonferroni
         - Holm-Bonferroni
         - Benjamini–Hochberg FDR
    5. Computes rank-biserial effect size for each pair
    6. Merges categories when they are:
         - NOT statistically different after correction
         - AND practically similar (|r| < r_thresh)

    Notes
    -----
    - Best used for exploratory grouping / feature engineering.
    - Connected-component merging may over-collapse due to transitive similarity.
      For publication-grade collapsing, stricter all-pair merge rules are preferred.
    """

    def __init__(
        self,
        duration_col="TransplantSurvivalDay",
        min_n=30,
        r_thresh=0.10,       # Effect size threshold for merging (rank-biserial correlation)
        method="bonferroni", # Multiple-testing correction method: "bonferroni", "holm", or "fdr"
        suppress_pairwise=False,
        plot=True,
        rotation=0
    ):
        """
        Initialize the ranker.

        Parameters
        ----------
        duration_col : str, default="TransplantSurvivalDay"
            Survival or duration outcome column.

        min_n : int, default=30
            Minimum category size required for inclusion.

        r_thresh : float, default=0.10
            Maximum absolute rank-biserial effect size allowed for collapsing.

        method : {"bonferroni", "holm", "fdr"}, default="bonferroni"
            Multiple-testing correction method for pairwise tests.

        suppress_pairwise : bool, default=False
            If True, suppress detailed pairwise comparison printing.

        plot : bool, default=True
            If True, display violin plot of survival distributions.

        rotation : int, default=0
            Rotation angle for x-axis category labels.
        """
        self.duration_col = duration_col
        self.min_n = min_n
        self.r_thresh = r_thresh
        self.method = method.lower()
        self.suppress_pairwise = suppress_pairwise
        self.plot = plot
        self.rotation = rotation

        if self.method not in {"bonferroni", "holm", "fdr"}:
            raise ValueError("method must be 'bonferroni', 'holm', or 'fdr'")

    # ------------------------------------------------------------------
    # Multiple-testing correction utilities
    # ------------------------------------------------------------------
    def _holm(self, pvals):
        """
        Apply Holm-Bonferroni correction.

        Parameters
        ----------
        pvals : array-like
            Raw p-values.

        Returns
        -------
        p_adj : np.ndarray
            Adjusted p-values.

        reject : np.ndarray of bool
            True if significant at alpha = 0.05.
        """
        pvals = np.asarray(pvals, dtype=float)
        m = len(pvals)

        order = np.argsort(pvals)
        sorted_p = pvals[order]

        adj_sorted = np.empty(m)
        for i in range(m):
            adj_sorted[i] = (m - i) * sorted_p[i]

        adj_sorted = np.maximum.accumulate(adj_sorted)
        adj_sorted = np.clip(adj_sorted, 0, 1)

        p_adj = np.empty(m)
        p_adj[order] = adj_sorted

        reject = p_adj < 0.05
        return p_adj, reject

    def _fdr_bh(self, pvals):
        """
        Apply Benjamini-Hochberg FDR correction.

        Parameters
        ----------
        pvals : array-like
            Raw p-values.

        Returns
        -------
        p_adj : np.ndarray
            FDR-adjusted p-values.

        reject : np.ndarray of bool
            True if significant at FDR alpha = 0.05.
        """
        pvals = np.asarray(pvals, dtype=float)
        m = len(pvals)

        order = np.argsort(pvals)
        sorted_p = pvals[order]

        adj_sorted = sorted_p * m / (np.arange(1, m + 1))
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
        adj_sorted = np.clip(adj_sorted, 0, 1)

        p_adj = np.empty(m)
        p_adj[order] = adj_sorted

        reject = p_adj < 0.05
        return p_adj, reject

    def _adjust_pvalues(self, raw_pvals):
        """
        Apply selected multiple-testing correction method.

        Parameters
        ----------
        raw_pvals : array-like
            Raw pairwise p-values.

        Returns
        -------
        p_adj : np.ndarray
            Adjusted p-values.

        reject : np.ndarray of bool
            Significance decisions.
        """
        raw_pvals = np.asarray(raw_pvals, dtype=float)

        if self.method == "holm":
            return self._holm(raw_pvals)
        elif self.method == "fdr":
            return self._fdr_bh(raw_pvals)
        else:  # bonferroni
            m = len(raw_pvals)
            p_adj = np.clip(raw_pvals * m, 0, 1)
            reject = p_adj < 0.05
            return p_adj, reject

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def fit(self, data, feature_col, custom_order=None):
        """
        Run survival-aware ranking and collapsing on a nominal feature.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset.

        feature_col : str
            Categorical feature to evaluate.

        custom_order : list, optional
            Optional category ordering for plotting/reporting.

        Returns
        -------
        dict or None
            Structured results dictionary.
        """
        # --------------------------------------------------------------
        # 1. Clean and filter data
        # --------------------------------------------------------------
        df = data.dropna(subset=[self.duration_col]).copy()
        df[feature_col] = df[feature_col].astype(str)

        counts = df[feature_col].value_counts()
        valid = counts[counts >= self.min_n].index.tolist()
        filtered = df[df[feature_col].isin(valid)].copy()

        if filtered.empty:
            print(f"[SKIPPED] No categories in {feature_col} have n ≥ {self.min_n}")
            return None

        present = filtered[feature_col].unique()
        if custom_order:
            order = [c for c in custom_order if c in present]
        else:
            order = sorted(present)

        if len(order) < 2:
            print(f"[WARNING] Cannot perform statistical comparison for {feature_col}")
            print(f"          Only 1 category has n ≥ {self.min_n}.")
            print(f"          You can only compare MORE THAN ONE category since n < min_n.")
            print()
            return {
                "feature": feature_col,
                "groups": [[order[0]]],
                "mapping": {order[0]: "Group_1"},
                "pairwise": [],
                "pairwise_df": pd.DataFrame(),
                "kruskal": {"H": None, "p": None, "eta2": None},
                "counts": counts.to_dict()
            }

        # --------------------------------------------------------------
        # 2. Build category → survival arrays
        # --------------------------------------------------------------
        group_map = {
            cat: grp[self.duration_col].dropna().values
            for cat, grp in filtered.groupby(feature_col)
        }
        groups = [group_map[c] for c in order]

        # --------------------------------------------------------------
        # 3. Global Kruskal–Wallis
        # --------------------------------------------------------------
        H, p = kruskal(*groups)
        k = len(groups)
        n = len(filtered)

        eta2 = (H - k + 1) / (n - k) if (n - k) > 0 else np.nan
        eta2 = max(0, eta2) if pd.notna(eta2) else np.nan

        # --------------------------------------------------------------
        # 4. Optional visualization
        # --------------------------------------------------------------
        if self.plot:
            plt.figure(figsize=(12, 8))
            sns.violinplot(
                data=filtered,
                x=feature_col,
                y=self.duration_col,
                order=order,
                hue=feature_col,
                palette="viridis",
                inner="quartile",
                legend=False
            )
            plt.title(f"{feature_col} — H={H:.2f}, η²={eta2:.3f}")
            ha = 'right' if self.rotation == 45 else 'center'
            plt.xticks(rotation=self.rotation, ha=ha)
            plt.tight_layout()
            plt.show()

        # --------------------------------------------------------------
        # 5. Pairwise Mann–Whitney U tests
        # --------------------------------------------------------------
        pairs = list(combinations(order, 2))
        raw_pvals = []
        pair_records = []

        for c1, c2 in pairs:
            g1, g2 = group_map[c1], group_map[c2]

            stat, p_pair = mannwhitneyu(g1, g2, alternative="two-sided")
            r = 1 - (2 * stat) / (len(g1) * len(g2))

            raw_pvals.append(p_pair)
            pair_records.append({
                "cat1": c1,
                "cat2": c2,
                "p_raw": p_pair,
                "r": r
            })

        raw_pvals = np.asarray(raw_pvals)

        # --------------------------------------------------------------
        # 6. Multiple-testing correction
        # --------------------------------------------------------------
        p_adj, reject = self._adjust_pvalues(raw_pvals)

        for i, row in enumerate(pair_records):
            row["p_adj"] = p_adj[i]
            row["sig"] = bool(reject[i])

        pairwise_df = pd.DataFrame(pair_records)

        # --------------------------------------------------------------
        # 7. Console reporting
        # --------------------------------------------------------------
        num_comparisons = len(pair_records)
        base_alpha = 0.05
        bonf_alpha = base_alpha / num_comparisons if num_comparisons > 0 else base_alpha

        print("=" * 80)
        print(f"RANKING REPORT: {feature_col} (n ≥ {self.min_n})")
        print("=" * 80)

        # --------------------------------------------------------------
        # INCLUDED CATEGORIES (those meeting min_n)
        # --------------------------------------------------------------
        included = counts[counts >= self.min_n]

        print("INCLUDED CATEGORIES:")
        for cat, n in included.sort_index().items():
            print(f" * {cat:<50}: {n:6d} records")

        # --------------------------------------------------------------
        # BYPASSED CATEGORIES (those with n < min_n)
        # --------------------------------------------------------------
        bypassed = counts[counts < self.min_n]

        if len(bypassed) > 0:
            print("BYPASSED CATEGORIES (n < min_n):")
            for cat, n in bypassed.sort_index().items():
                print(f" * {cat:<50}: {n:6d} records")
        else:
            print("BYPASSED CATEGORIES: None")

        print("-" * 80)
        print(f"Kruskal H: {H:.4f} | P-value: {p:.6e} | Effect η²: {eta2:.4f}")
        print(f"Correction Method: {self.method.upper()}")
        
        if self.method == "bonferroni":
            print(f"Pairwise Testing (Bonferroni Alpha: {bonf_alpha:.4f} | r_threshold: {self.r_thresh:.2f})")
        else:
            print(f"Pairwise Testing (Base Alpha: {base_alpha} | Method: {self.method.upper()} | r_threshold: {self.r_thresh:.2f})")
        print("-" * 80)

        if not self.suppress_pairwise:
            # For Holm, we need to know the rank to show the threshold
            # Sort records by raw p-value for display/logic clarity
            sorted_records = sorted(pair_records, key=lambda x: x['p_raw'])
            
            for i, row in enumerate(sorted_records):
                # Calculate the specific threshold for this step if using Holm
                # Step Alpha = Alpha / (m - rank + 1)
                if self.method == "holm":
                    current_threshold = base_alpha / (num_comparisons - i)
                    threshold_label = f"thr={current_threshold:.4f}"
                elif self.method == "bonferroni":
                    threshold_label = f"thr={bonf_alpha:.4f}"
                else:
                    threshold_label = f"fdr_alpha={base_alpha}"

                status = "*Sig*" if row['sig'] else "Not Sig"
                
                print(
                    f"{row['cat1']:<12} vs {row['cat2']:<12} | "
                    f"p={row['p_raw']:.4e} | {threshold_label} | "
                    f"r={row['r']:>6.3f} | {status}"
                )
        else:
            print("[Pairwise comparison output suppressed]")

        print("=" * 80)

        # --------------------------------------------------------------
        # 8. Graph-based collapsing
        # --------------------------------------------------------------
        G = nx.Graph()
        G.add_nodes_from(order)

        for row in pair_records:
            if (not row["sig"]) and abs(row["r"]) < self.r_thresh:
                G.add_edge(row["cat1"], row["cat2"])

        collapsed_groups = [
            sorted(list(comp)) for comp in nx.connected_components(G)
        ]

        # --------------------------------------------------------------
        # 9. Add small categories as overflow group
        # --------------------------------------------------------------
        small_cats = sorted(counts[counts < self.min_n].index.tolist())

        # --------------------------------------------------------------
        # 10. Build mapping for included categories (collapsed groups)
        # --------------------------------------------------------------
        mapping = {}

        # Assign real collapsed groups
        for group_id, group in enumerate(collapsed_groups, start=1):
            for cat in group:
                mapping[cat] = f"Group_{group_id}"

        # Assign bypassed categories to 'Other'
        for cat in small_cats:
            mapping[cat] = "Other"

        # --------------------------------------------------------------
        # Print number of collapsed groups
        # --------------------------------------------------------------
        num_groups = len(collapsed_groups)
        print(f"::::: Collapsed into {num_groups} groups (excluding 'Other') :::::")
        print("=" * 80 + "\n")
       
        # --------------------------------------------------------------
        # 11. Return structured result
        # --------------------------------------------------------------               
        return {
            "feature": feature_col,
            "method_used": self.method,
            "groups": collapsed_groups,
            "mapping": mapping,
            "pairwise": pair_records,
            "pairwise_df": pairwise_df,
            "kruskal": {"H": H, "p": p, "eta2": eta2},
            "counts": counts.to_dict(),
            
        }
    # ------------------------------------------------------------------
    # Helper: Apply learned mapping to a dataframe
    # ------------------------------------------------------------------
    def transform(self, data, feature_col, mapping, new_col=None):
        """
        Apply collapsed category mapping to a dataframe.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe.

        feature_col : str
            Original categorical column.

        mapping : dict
            Category → collapsed group mapping from fit() output.

        new_col : str, optional
            Name of new output column. Defaults to <feature_col>_collapsed.

        Returns
        -------
        pd.DataFrame
            DataFrame with collapsed feature added.
        """
        df = data.copy()
        if new_col is None:
            new_col = f"{feature_col}_collapsed"

        df[new_col] = df[feature_col].astype(str).map(mapping).fillna("Unmapped")
        return df
    

    def build_feature_dict(self,results):
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

        # --------------------------------------------------------------
        # Separate real groups ("Group_X") from non-group buckets
        # --------------------------------------------------------------
        real_groups = {k: v for k, v in groups.items() if k.startswith("Group_")}
        other_groups = {k: v for k, v in groups.items() if not k.startswith("Group_")}

        # --------------------------------------------------------------
        # Sort real groups by numeric suffix
        # --------------------------------------------------------------
        sorted_real = dict(sorted(
            real_groups.items(),
            key=lambda x: int(x[0].split("_")[1])
        ))

        # --------------------------------------------------------------
        # Append non-group buckets at the end
        # --------------------------------------------------------------
        sorted_groups = {**sorted_real, **other_groups}

        return {
            "feature": feature,
            "method_used": method_used,
            **sorted_groups
        }

    # ------------------------------------------------------------------
    # Convert results dict → long-format DataFrame
    # ------------------------------------------------------------------
    def to_long_dataframe(self, results):
        """
        Convert a structured statistical analysis result into a standardized
        long-format DataFrame.

        Supports:
            - feature
            - method_used
            - groups
            - mapping
            - pairwise (p_raw, p_adj, r, sig)
            - kruskal stats
            - counts
            - pairwise_df (ignored unless needed)

        Returns
        -------
        pd.DataFrame
            Long-format table with columns:
                section | key | value
        """

        rows = []

        # -------------------------
        # Feature name
        # -------------------------
        rows.append({
            "section": "feature",
            "key": "feature",
            "value": results["feature"]
        })

        # -------------------------
        # Method name
        # -------------------------
        rows.append({
            "section": "method",
            "key": "method",
            "value": results.get("method_used", "unknown")
        })

        # -------------------------
        # Collapsed groups
        # -------------------------
        for i, group in enumerate(results["groups"], start=1):
            rows.append({
                "section": "groups",
                "key": f"Group_{i}",
                "value": group
            })

        # -------------------------
        # Mapping (category → group)
        # -------------------------
        for cat, grp in results["mapping"].items():
            rows.append({
                "section": "mapping",
                "key": cat,
                "value": grp
            })

        # -------------------------
        # Pairwise results
        # -------------------------
        for row in results["pairwise"]:
            rows.append({
                "section": "pairwise",
                "key": f"{row['cat1']} vs {row['cat2']}",
                "value": {
                    "p_raw": float(row.get("p_raw", np.nan)),
                    "p_adj": float(row.get("p_adj", np.nan)),
                    "r": float(row.get("r", np.nan)),
                    "sig": bool(row.get("sig", False))
                }
            })

        # -------------------------
        # Global Kruskal–Wallis stats
        # -------------------------
        for k, v in results["kruskal"].items():
            rows.append({
                "section": "kruskal",
                "key": k,
                "value": float(v)
            })

        # -------------------------
        # Category counts
        # -------------------------
        for cat, n in results["counts"].items():
            rows.append({
                "section": "counts",
                "key": cat,
                "value": int(n)
            })

        return pd.DataFrame(rows)