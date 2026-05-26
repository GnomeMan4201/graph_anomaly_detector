"""
GraphBuilder
────────────
Builds a directed, weighted NetworkX graph from a canonical interactions
DataFrame.  Edge weight = interaction frequency between that (src, dst) pair.

Temporal slicing returns a subgraph restricted to events within a rolling
window ending at `end_ts`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import networkx as nx
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


class GraphBuilder:
    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame]  = None
        self._G:  Optional[nx.DiGraph]    = None

    # ── public api ────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build the full interaction graph from a canonical DataFrame.
        Edge attribute  ``weight``  = co-occurrence count.
        Node attribute  ``events``  = total out-events for that node.
        """
        self._df = df
        G = nx.DiGraph()

        # Add all unique users as nodes (both sides of interactions)
        all_nodes = pd.unique(df[["user_id", "target_id"]].values.ravel())
        G.add_nodes_from(all_nodes)

        # Count edges
        edge_counts = (
            df.groupby(["user_id", "target_id"])
            .size()
            .reset_index(name="weight")
        )
        for _, row in edge_counts.iterrows():
            G.add_edge(row["user_id"], row["target_id"], weight=int(row["weight"]))

        # Node-level event totals
        node_events = df.groupby("user_id").size().to_dict()
        nx.set_node_attributes(G, node_events, "events")

        log.info(
            "graph built: nodes=%d  edges=%d  density=%.6f",
            G.number_of_nodes(),
            G.number_of_edges(),
            nx.density(G),
        )
        self._G = G
        return G

    def slice_window(
        self,
        df: pd.DataFrame,
        end_ts: float,
        window_seconds: int,
    ) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """
        Return (G_slice, df_slice) restricted to events in
        [end_ts - window_seconds, end_ts].
        """
        start_ts = end_ts - window_seconds
        mask     = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
        df_slice = df[mask].copy()
        if df_slice.empty:
            return nx.DiGraph(), df_slice
        G_slice = self.build(df_slice)
        return G_slice, df_slice

    def get_graph(self) -> nx.DiGraph:
        if self._G is None:
            raise RuntimeError("No graph built yet. Call build() first.")
        return self._G

    # ── convenience ───────────────────────────────────────────────────────────

    @staticmethod
    def to_undirected_weighted(G: nx.DiGraph) -> nx.Graph:
        """Collapse to undirected graph, summing edge weights."""
        return G.to_undirected(reciprocal=False)

    @staticmethod
    def subgraph_for_nodes(G: nx.DiGraph, nodes: list) -> nx.DiGraph:
        return G.subgraph(nodes).copy()
