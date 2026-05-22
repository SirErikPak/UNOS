import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from scipy.stats import kruskal, mannwhitneyu
from typing import List, Dict, Any, Optional

# Elegant presentation layer engines
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.box import ROUNDED
from rich.text import Text


class NominalSurvivalRanker:
    """
    Rank and collapse nominal categorical levels based on survival outcome similarity.
    Optimized with premium Rich diagnostics and modern Matplotlib configurations.
    """

    def __init__(
        self,
        duration_col: str = "TransplantSurvivalDay",
        min_n: int = 30,
        r_thresh: float = 0.10,
        method: str = "bonferroni",
        suppress_pairwise: bool = False,
        plot: bool = True,
        rotation: int = 0,
        ignore_sig_on_low_effect: bool = True  # Allows collapsing when effect size is small despite high sample power
    ):
        self.duration_col = duration_col
        self.min_n = min_n
        self.r_thresh = r_thresh
        self.method = method.lower()
        self.suppress_pairwise = suppress_pairwise
        self.plot = plot
        self.rotation = rotation
        self.ignore_sig_on_low_effect = ignore_sig_on_low_effect
        self.console = Console()

        if self.method not in {"bonferroni", "holm", "fdr"}:
            raise ValueError("method must be 'bonferroni', 'holm', or 'fdr'")

    def _holm(self, pvals: np.ndarray):
        m = len(pvals)
        order = np.argsort(pvals)
        sorted_p = pvals[order]
        adj_sorted = np.clip(np.arange(m, 0, -1) * sorted_p, 0, 1)
        adj_sorted = np.maximum.accumulate(adj_sorted)
        p_adj = np.empty(m)
        p_adj[order] = adj_sorted
        return p_adj, p_adj < 0.05

    def _fdr_bh(self, pvals: np.ndarray):
        m = len(pvals)
        order = np.argsort(pvals)
        sorted_p = pvals[order]
        adj_sorted = np.clip(sorted_p * m / np.arange(1, m + 1), 0, 1)
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
        p_adj = np.empty(m)
        p_adj[order] = adj_sorted
        return p_adj, p_adj < 0.05

    def _adjust_pvalues(self, raw_pvals: np.ndarray):
        if self.method == "holm":
            return self._holm(raw_pvals)
        elif self.method == "fdr":
            return self._fdr_bh(raw_pvals)
        else:
            m = len(raw_pvals)
            return np.clip(raw_pvals * m, 0, 1), (raw_pvals * m) < 0.05

    def fit(self, data: pd.DataFrame, feature_col: str, custom_order: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        # 1. Clean data framework
        df = data.dropna(subset=[self.duration_col]).copy()
        df[feature_col] = df[feature_col].astype(str)

        counts = df[feature_col].value_counts()
        valid = counts[counts >= self.min_n].index.tolist()
        filtered = df[df[feature_col].isin(valid)].copy()

        if filtered.empty:
            self.console.print(Panel(f"[bold red]✕ Skipped:[/bold red] No categories in [yellow]{feature_col}[/yellow] meet n ≥ {self.min_n}", box=ROUNDED, border_style="red"))
            return None

        present = filtered[feature_col].unique()
        order = [c for c in custom_order if c in present] if custom_order else sorted(present)

        if len(order) < 2:
            return {
                "feature": feature_col, "groups": [[order[0]]] if order else [],
                "mapping": {order[0]: "Group_1"} if order else {}, "pairwise": [],
                "pairwise_df": pd.DataFrame(), "kruskal": {"H": None, "p": None, "eta2": None}, "counts": counts.to_dict()
            }

        # 2. Extract arrays
        group_map = {cat: grp.values for cat, grp in filtered.groupby(feature_col)[self.duration_col]}
        groups = [group_map[c] for c in order]

        # 3. Global test calculation
        H, p_kruskal = kruskal(*groups)
        k, n = len(groups), len(filtered)
        eta2 = max(0, (H - k + 1) / (n - k)) if (n - k) > 0 else np.nan

        # 4. Modern Polished Violin Visualization
        if self.plot:
            sns.set_theme(style="whitegrid", palette="muted")
            fig, ax = plt.subplots(figsize=(10, 5.5))
            
            sns.violinplot(
                data=filtered, x=feature_col, y=self.duration_col, order=order,
                hue=feature_col, palette="crest", inner="quartile",
                linewidth=1.2, density_norm="width", legend=False, ax=ax
            )
            
            # Clean minimalistic borders
            sns.despine(left=True, bottom=True)
            ax.set_title(f"{feature_col} Survival Profiles\n(Omnibus H: {H:.2f}, η²: {eta2:.3f})", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel("Categorical Variable Levels", fontsize=10, labelpad=10)
            ax.set_ylabel(self.duration_col, fontsize=10, labelpad=10)
            
            # Warning-free Tick assignment for Matplotlib 3.10+
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=self.rotation, ha='right' if self.rotation else 'center', fontsize=9)
            ax.tick_params(colors="#4f4f4f", labelsize=9)
            
            fig.tight_layout()
            plt.show()

        # 5. Pairwise Mann-Whitney evaluations
        pairs = list(combinations(order, 2))
        pair_records, raw_pvals = [], []

        for c1, c2 in pairs:
            g1, g2 = group_map[c1], group_map[c2]
            stat, p_pair = mannwhitneyu(g1, g2, alternative="two-sided")
            r = (2.0 * stat / (len(g1) * len(g2))) - 1.0  # Normalized framework bounds

            raw_pvals.append(p_pair)
            pair_records.append({"cat1": c1, "cat2": c2, "p_raw": p_pair, "r": r})

        # 6. Apply multi-testing correction passes
        p_adj, reject = self._adjust_pvalues(np.asarray(raw_pvals))
        for idx, row in enumerate(pair_records):
            row["p_adj"] = p_adj[idx]
            row["sig"] = bool(reject[idx])

        pairwise_df = pd.DataFrame(pair_records)

        # 7. Modernized Ultra-Clean Rich Board Report
        title_text = Text.from_markup(f"[bold white]✦ Ranking Summary:[/bold white] [cyan]{feature_col}[/cyan]")
        title_text.justify = "left"
        
        report_table = Table(title=title_text, box=ROUNDED, border_style="dim", header_style="bold magenta", expand=False)
        report_table.add_column("Comparison Pair", justify="left", style="white")
        report_table.add_column("Raw p-value", justify="right", style="yellow")
        report_table.add_column("Adj p-value", justify="right", style="cyan")
        report_table.add_column("Effect Size (r)", justify="right", style="green")
        report_table.add_column("Status", justify="center")

        if not self.suppress_pairwise:
            for r in pair_records:
                status_badge = "[bold red]Significant[/bold red]" if r["sig"] else "[dim white]Not Sig[/dim white]"
                report_table.add_row(
                    f"{r['cat1']} [dim]vs[/dim] {r['cat2']}",
                    f"{r['p_raw']:.4e}",
                    f"{r['p_adj']:.4e}",
                    f"{r['r']:+.3f}",
                    status_badge
                )
        
        self.console.print(report_table)

        # 8. Graph Partitioning (With Sample Power Constraint Guard)
        G = nx.Graph()
        G.add_nodes_from(order)
        for row in pair_records:
            # Check proximity based on configuration rules
            if self.ignore_sig_on_low_effect:
                # Disregard p-value significance if effect size is small
                should_merge = abs(row["r"]) < self.r_thresh
            else:
                # Traditional strict route: requires not significant AND small effect
                should_merge = (not row["sig"]) and (abs(row["r"]) < self.r_thresh)
                
            if should_merge:
                G.add_edge(row["cat1"], row["cat2"])

        collapsed_groups = [sorted(list(comp)) for comp in nx.connected_components(G)]

        # 9. Formulate output mappings
        small_cats = sorted(counts[counts < self.min_n].index.tolist())
        mapping = {}
        for g_idx, group in enumerate(collapsed_groups, start=1):
            for cat in group:
                mapping[cat] = f"Group_{g_idx}"
        for cat in small_cats:
            mapping[cat] = "Other"

        # Print quick grouping result card
        self.console.print(Panel(f"[bold green]✔ Optimization Complete![/bold green] Structured items collapsed into [bold cyan]{len(collapsed_groups)}[/bold cyan] groups (excluding 'Other' bucket).", box=ROUNDED, border_style="green", expand=False))
        self.console.print()

        return {
            "feature": feature_col, "method_used": self.method, "groups": collapsed_groups,
            "mapping": mapping, "pairwise": pair_records, "pairwise_df": pairwise_df,
            "kruskal": {"H": H, "p": p_kruskal, "eta2": eta2}, "counts": counts.to_dict(),
        }

    def transform(self, data: pd.DataFrame, feature_col: str, mapping: dict, new_col: Optional[str] = None) -> pd.DataFrame:
        df = data.copy()
        out_name = new_col if new_col else f"{feature_col}_collapsed"
        df[out_name] = df[feature_col].astype(str).map(mapping).fillna("Unmapped")
        return df

    def build_feature_dict(self, results: dict) -> dict:
        feature = results["feature"]
        mapping = results["mapping"]
        counts = results["counts"]
        method_used = results.get("method_used", "unknown")

        rev = defaultdict(list)
        for cat, grp in mapping.items():
            rev[grp].append(cat)

        sorted_groups = {}
        real_keys = sorted([k for k in rev.keys() if k.startswith("Group_")], key=lambda x: int(x.split("_")[1]))
        other_keys = [k for k in rev.keys() if not k.startswith("Group_")]

        for k in (real_keys + other_keys):
            cats = sorted(rev[k])
            sorted_groups[k] = {
                "Total_N": int(sum(int(counts.get(c, 0)) for c in cats)),
                "Categories": cats
            }

        return {"feature": feature, "method_used": method_used, **sorted_groups}

    def to_long_dataframe(self, results: dict) -> pd.DataFrame:
        rows = [
            {"section": "feature", "key": "feature", "value": results["feature"]},
            {"section": "method", "key": "method", "value": results.get("method_used", "unknown")}
        ]
        for i, group in enumerate(results["groups"], start=1):
            rows.append({"section": "groups", "key": f"Group_{i}", "value": group})
        for cat, grp in results["mapping"].items():
            rows.append({"section": "mapping", "key": cat, "value": grp})
        for r in results["pairwise"]:
            rows.append({
                "section": "pairwise",
                "key": f"{r['cat1']} vs {r['cat2']}",
                "value": {"p_raw": float(r["p_raw"]), "p_adj": float(r["p_adj"]), "r": float(r["r"]), "sig": bool(r["sig"])}
            })
        for k, v in results["kruskal"].items():
            rows.append({"section": "kruskal", "key": k, "value": float(v) if v is not None else np.nan})
        for cat, n in results["counts"].items():
            rows.append({"section": "counts", "key": cat, "value": int(n)})

        return pd.DataFrame(rows)