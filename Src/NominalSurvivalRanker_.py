import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from scipy.stats import kruskal, mannwhitneyu

# NOTE: If you have right-censored data, 'pip install lifelines' 
# and uncomment the log-rank logic below.
# from lifelines.statistics import logrank_test 

class NominalSurvivalRanker:
    def __init__(
        self,
        duration_col="TransplantSurvivalDay",
        event_col=None,  # Set this if you have a 0/1 column for censoring
        min_n=30,
        r_thresh=0.10,
        method="holm",
        suppress_pairwise=False,
        plot=True,
        rotation=45,
        verbose=True
    ):
        self.duration_col = duration_col
        self.event_col = event_col
        self.min_n = min_n
        self.r_thresh = r_thresh
        self.method = method.lower()
        self.suppress_pairwise = suppress_pairwise
        self.plot = plot
        self.rotation = rotation
        self.verbose = verbose
        self.results_ = None

    # ============================================================
    # Multiple Testing Corrections
    # ============================================================

    def _adjust_pvalues(self, pvals):
        pvals = np.asarray(pvals, dtype=float)
        m = len(pvals)
        if m == 0: return pvals, []

        if self.method == "bonferroni":
            p_adj = np.clip(pvals * m, 0, 1)
        
        elif self.method == "holm":
            order = np.argsort(pvals)
            p_sorted = pvals[order]
            adj_sorted = p_sorted * np.arange(m, 0, -1)
            adj_sorted = np.maximum.accumulate(adj_sorted)
            p_adj = np.empty(m)
            p_adj[order] = np.clip(adj_sorted, 0, 1)
            
        elif self.method == "fdr":
            order = np.argsort(pvals)
            p_sorted = pvals[order]
            adj_sorted = p_sorted * m / np.arange(1, m + 1)
            adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
            p_adj = np.empty(m)
            p_adj[order] = np.clip(adj_sorted, 0, 1)
            
        reject = p_adj < 0.05
        return p_adj, reject

    # ============================================================
    # Grouping Logic
    # ============================================================

    def _build_strict_groups(self, categories, pairwise_lookup):
        """
        Groups categories such that every member in a group is NOT 
        significantly different from EVERY other member in that group.
        """
        groups = []
        for cat in categories:
            placed = False
            for group in groups:
                # Check compatibility with all existing members of the group
                is_compatible = True
                for existing_member in group:
                    pair = tuple(sorted((cat, existing_member)))
                    record = pairwise_lookup.get(pair)
                    
                    # Reject group if pair is statistically different OR has high effect size
                    if record and (record["sig"] or record["r_abs"] >= self.r_thresh):
                        is_compatible = False
                        break
                
                if is_compatible:
                    group.append(cat)
                    placed = True
                    break
            
            if not placed:
                groups.append([cat])
        return groups

    # ============================================================
    # Core Engine
    # ============================================================

    def fit(self, data, feature_col, custom_order=None):
        # 1. Cleaning & Filtering
        df = data.dropna(subset=[self.duration_col]).copy()
        counts = df[feature_col].value_counts()
        valid_cats = counts[counts >= self.min_n].index.tolist()
        filtered = df[df[feature_col].isin(valid_cats)].copy()

        if filtered.empty:
            if self.verbose: print(f"[SKIPPED] No categories meet min_n={self.min_n}")
            return None

        # 2. Setup Comparison Order
        present = filtered[feature_col].unique()
        order = [c for c in (custom_order or sorted(present)) if c in present]
        
        # 3. Global Test (Kruskal-Wallis)
        group_data = [filtered[filtered[feature_col] == c][self.duration_col].values for c in order]
        H, p_global = kruskal(*group_data)
        n_total, k_groups = len(filtered), len(order)
        epsilon2 = max(0, (H - k_groups + 1) / (n_total - k_groups)) if n_total > k_groups else 0

        # 4. Pairwise Tests
        pairs = list(combinations(order, 2))
        pair_records = []
        for c1, c2 in pairs:
            s1 = filtered[filtered[feature_col] == c1][self.duration_col]
            s2 = filtered[filtered[feature_col] == c2][self.duration_col]
            
            # Use Mann-Whitney U for rank-based comparison
            stat, p_raw = mannwhitneyu(s1, s2, alternative="two-sided")
            r = (2 * stat) / (len(s1) * len(s2)) - 1 # Simple effect size
            
            pair_records.append({"cat1": c1, "cat2": c2, "p_raw": p_raw, "r": r, "r_abs": abs(r)})

        # 5. Corrections & Grouping
        p_adj, reject = self._adjust_pvalues([x["p_raw"] for x in pair_records])
        for i, rec in enumerate(pair_records):
            rec["p_adj"], rec["sig"] = p_adj[i], bool(reject[i])

        lookup = {tuple(sorted((r["cat1"], r["cat2"]))): r for r in pair_records}
        collapsed_groups = self._build_strict_groups(order, lookup)

        # 6. Final Mapping
        mapping = {cat: f"RankGroup_{i+1}" for i, grp in enumerate(collapsed_groups) for cat in grp}
        for cat in counts[counts < self.min_n].index:
            mapping[cat] = "Small_N_Unranked"

        self.results_ = {
            "feature": feature_col,
            "mapping": mapping,
            "stats": {"H": H, "p_global": p_global, "epsilon2": epsilon2},
            "pairwise": pair_records
        }

        if self.plot:
            self._plot_results(filtered, feature_col, order, H, epsilon2)

        return self.results_

    def _plot_results(self, df, feature, order, H, e2):
        plt.figure(figsize=(max(10, len(order)*0.8), 6))
        sns.boxplot(data=df, x=feature, y=self.duration_col, order=order, palette="coolwarm", showfliers=False)
        sns.stripplot(data=df, x=feature, y=self.duration_col, order=order, color="black", alpha=0.2, size=3)
        plt.title(f"Outcome Distribution by {feature}\nGlobal H: {H:.2f} (ε²: {e2:.3f})")
        plt.xticks(rotation=self.rotation)
        plt.tight_layout()
        plt.show()

    def transform(self, data, feature_col):
        if not self.results_: raise ValueError("Model must be fitted first.")
        new_col = f"{feature_col}_ranked"
        return data[feature_col].map(self.results_["mapping"]).fillna("Unmapped")