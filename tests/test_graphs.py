from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_loader import load_data
from graphs import (
    build_graph_citation_network,
    build_graph_most_cited,
    build_graph_top_citing,
    graph_to_plotly,
)


def test_build_graph_shapes_and_types() -> None:
    papers = pd.DataFrame(
        [
            {"Unique ID": 1, "Cites": 10, "Title": "Base", "Authors": "A, B", "Year": 2020, "Source": "RF"},
            {"Unique ID": 2, "Cites": 15, "Title": "Citer", "Authors": "C", "Year": 2021, "Source": "J1"},
        ]
    )
    edges = pd.DataFrame(
        [{"Citing Paper Unique ID": 2, "Cited Paper Unique ID": 1}]
    )

    g = build_graph_citation_network(papers, edges)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    assert g.nodes[1]["node_type"] == "cited"
    assert g.nodes[2]["node_type"] == "citing"


def test_graph_to_plotly_deterministic_layout() -> None:
    df = pd.DataFrame(
        [
            {"Cites": 10, "Authors": "A", "Title": "P1", "Year": 2020, "Source": "S"},
            {"Cites": 20, "Authors": "B", "Title": "P2", "Year": 2021, "Source": "S"},
        ]
    )
    g = build_graph_most_cited(df)

    state = {"layout_seed": 24, "node_size_scale": 1.0}
    fig1 = graph_to_plotly(g, "most_cited", state)
    fig2 = graph_to_plotly(g, "most_cited", state)

    assert list(fig1.data[1]["x"]) == list(fig2.data[1]["x"])
    assert list(fig1.data[1]["y"]) == list(fig2.data[1]["y"])
    assert len(fig1.data[1]["customdata"]) == 2


def test_top_citing_grouping() -> None:
    df = pd.DataFrame(
        [
            {"Cites": 5, "Source": "Journal A", "Title": "X", "Year": 2019},
            {"Cites": 15, "Source": "Journal A", "Title": "Y", "Year": 2020},
            {"Cites": 8, "Source": "Journal B", "Title": "Z", "Year": 2021},
        ]
    )
    g = build_graph_top_citing(df)
    assert g.number_of_nodes() == 2


def test_real_data_counts_if_accessible() -> None:
    root = Path(__file__).resolve().parent.parent
    settings = root / "settings.toml"
    if not settings.exists():
        pytest.skip("settings.toml missing")

    import tomllib

    config = tomllib.loads(settings.read_text(encoding="utf-8-sig")).get("data", {})
    config["fallback_data_dir"] = tomllib.loads(settings.read_text(encoding="utf-8-sig")).get("cloud", {}).get(
        "fallback_data_dir", "data"
    )

    data = load_data(config)
    if data["unified_papers"].empty or data["edges"].empty:
        pytest.skip("real data not accessible from test environment")

    g = build_graph_citation_network(data["unified_papers"], data["edges"])

    assert len(data["unified_papers"]) == 84
    assert len(data["edges"]) == 63
    assert data["edges"]["Cited Paper Unique ID"].nunique() == 21
    assert g.number_of_edges() == 63
