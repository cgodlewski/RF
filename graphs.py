from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go
import pandas as pd


def format_authors(authors_string: str) -> str:
    if not isinstance(authors_string, str) or not authors_string.strip():
        return ""
    authors_list = [a.strip() for a in authors_string.split(",") if a.strip()]
    if len(authors_list) > 2:
        return f"{authors_list[0]} et al."
    if len(authors_list) == 2:
        return f"{authors_list[0]} & {authors_list[1]}"
    return authors_list[0]


def build_graph_most_cited(df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for _, row in df.iterrows():
        node_id = f"paper::{row.get('Title', '')}::{row.get('Year', '')}"
        title = str(row.get("Title", ""))
        authors = str(row.get("Authors", ""))
        year = int(row.get("Year", 0))
        cites = int(row.get("Cites", 0))
        hover = f"<b>{title}</b><br>{format_authors(authors)} ({year})<br>{cites} citations"
        graph.add_node(
            node_id,
            size=max(cites, 1),
            color_value=max(cites, 1),
            hover=hover,
            node_type="paper",
        )
    return graph


def build_graph_top_citing(df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    grouped = (
        df.groupby("Source", dropna=True)
        .agg(max_cites=("Cites", "max"), papers=("Title", "count"), latest_year=("Year", "max"))
        .reset_index()
    )
    for _, row in grouped.iterrows():
        source = str(row.get("Source", ""))
        cites = int(row.get("max_cites", 0))
        papers = int(row.get("papers", 0))
        latest_year = int(row.get("latest_year", 0))
        hover = (
            f"<b>{source}</b><br>Top citing-paper citations: {cites}"
            f"<br>Rows in top-3 dataset: {papers}<br>Latest year: {latest_year}"
        )
        graph.add_node(
            f"source::{source}",
            size=max(cites, 1),
            color_value=max(cites, 1),
            hover=hover,
            node_type="source",
        )
    return graph


def build_graph_citation_network(papers_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()

    cited_ids = set(edges_df["Cited Paper Unique ID"].astype(int).tolist()) if not edges_df.empty else set()
    citing_ids = set(edges_df["Citing Paper Unique ID"].astype(int).tolist()) if not edges_df.empty else set()

    for _, row in papers_df.iterrows():
        uid = int(row["Unique ID"])
        cites = int(row.get("Cites", 0))
        title = str(row.get("Title", ""))
        authors = str(row.get("Authors", ""))
        year = int(row.get("Year", 0))
        source = str(row.get("Source", ""))

        if uid in cited_ids:
            node_type = "cited"
            hover = f"<b>{title}</b><br>{format_authors(authors)} ({year})<br>{cites} citations"
        elif uid in citing_ids:
            node_type = "citing"
            hover = f"<b>{source}</b><br>{cites} citations"
        else:
            continue

        graph.add_node(
            uid,
            size=max(cites, 1),
            color_value=max(cites, 1),
            hover=hover,
            node_type=node_type,
        )

    for _, edge in edges_df.iterrows():
        citing_id = int(edge["Citing Paper Unique ID"])
        cited_id = int(edge["Cited Paper Unique ID"])
        if graph.has_node(citing_id) and graph.has_node(cited_id):
            graph.add_edge(citing_id, cited_id)

    return graph


def graph_to_plotly(G: nx.Graph, view_mode: str, ui_state: dict) -> go.Figure:
    seed = int(ui_state.get("layout_seed", 24))
    node_scale = float(ui_state.get("node_size_scale", 1.0))

    pos = nx.spring_layout(G, seed=seed) if G.number_of_nodes() > 0 else {}

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace["x"] += (x0, x1, None)
        edge_trace["y"] += (y0, y1, None)

    marker_sizes = []
    marker_colors = []
    marker_opacity = []
    x_vals = []
    y_vals = []
    hover_text = []
    customdata = []

    for node in G.nodes():
        x, y = pos[node]
        attrs = G.nodes[node]
        base_size = max(float(attrs.get("size", 1)), 1.0)

        x_vals.append(x)
        y_vals.append(y)
        hover_text.append(attrs.get("hover", str(node)))
        customdata.append(str(node))

        if view_mode == "citation_network":
            if attrs.get("node_type") == "citing":
                marker_colors.append("orange")
                marker_opacity.append(0.25)
            else:
                marker_colors.append("firebrick")
                marker_opacity.append(0.85)
            marker_sizes.append(base_size * node_scale)
        else:
            marker_colors.append(float(attrs.get("color_value", base_size)))
            marker_opacity.append(0.85)
            marker_sizes.append(base_size * node_scale)

    showscale = view_mode != "citation_network"
    colorbar = dict(thickness=15, title="# Citations", xanchor="left") if showscale else None

    node_trace = go.Scatter(
        x=x_vals,
        y=y_vals,
        text=hover_text,
        customdata=customdata,
        mode="markers",
        hovertemplate="%{text}<extra></extra>",
        marker=dict(
            showscale=showscale,
            colorscale="OrRd",
            color=marker_colors,
            size=marker_sizes,
            opacity=marker_opacity,
            sizemode="diameter",
            sizemin=4,
            colorbar=colorbar,
            line=dict(width=1),
        ),
    )

    titles = {
        "most_cited": "Most Cited Papers in Revue Finance",
        "top_citing": "Top Citing Journals/Papers",
        "citation_network": "Citation Network: Revue Finance and Citing Papers",
    }

    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=titles.get(view_mode, "Citation Graph"),
            showlegend=False,
            hovermode="closest",
            autosize=True,
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
