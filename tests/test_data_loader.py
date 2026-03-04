from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_data


def write_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_data_missing_file(tmp_path: Path) -> None:
    config = {
        "most_cited_csv": str(tmp_path / "missing1.csv"),
        "top_citing_csv": str(tmp_path / "missing2.csv"),
        "unified_papers_csv": str(tmp_path / "missing3.csv"),
        "edges_csv": str(tmp_path / "missing4.csv"),
    }
    data = load_data(config)
    assert data["most_cited"].empty
    assert len(data["_warnings"]) == 4
    assert "File not found" in data["_warnings"][0]


def test_load_data_missing_required_columns(tmp_path: Path) -> None:
    most = tmp_path / "most.csv"
    top = tmp_path / "top.csv"
    unified = tmp_path / "unified.csv"
    edges = tmp_path / "edges.csv"

    write_csv(most, "Cites,Authors,Title,Year\n10,A,Paper,2020\n")
    write_csv(top, "Cites,Source,Title\n20,Journal,P\n")
    write_csv(unified, "Cites,Authors,Title,Year,Source\n10,A,P,2020,S\n")
    write_csv(edges, "Citing Paper Unique ID\n2\n")

    data = load_data(
        {
            "most_cited_csv": str(most),
            "top_citing_csv": str(top),
            "unified_papers_csv": str(unified),
            "edges_csv": str(edges),
        }
    )

    assert data["most_cited"].empty
    assert data["top_citing"].empty
    assert data["unified_papers"].empty
    assert data["edges"].empty
    assert len(data["_warnings"]) == 4


def test_load_data_numeric_coercion(tmp_path: Path) -> None:
    most = tmp_path / "most.csv"
    top = tmp_path / "top.csv"
    unified = tmp_path / "unified.csv"
    edges = tmp_path / "edges.csv"

    write_csv(most, "Cites,Authors,Title,Year,Source\n11,a1 * ,Paper1,2020,Rev.\n")
    write_csv(top, "Cites,Source,Title,Year\n33,Journal A,TPaper,2021\n")
    write_csv(
        unified,
        "Cites,Authors,Title,Year,Source,Unique ID\n5,a2 * ,Base,2020,S1,1\n8,b3* ,Citing,2021,S2,2\n",
    )
    write_csv(edges, "Citing Paper Unique ID,Cited Paper Unique ID\n2,1\n")

    data = load_data(
        {
            "most_cited_csv": str(most),
            "top_citing_csv": str(top),
            "unified_papers_csv": str(unified),
            "edges_csv": str(edges),
        }
    )

    assert data["_warnings"] == []
    assert pd.api.types.is_integer_dtype(data["most_cited"]["Cites"])
    assert pd.api.types.is_integer_dtype(data["unified_papers"]["Unique ID"])
    assert data["most_cited"].iloc[0]["Authors"] == "a"


def test_load_data_fallback_directory(tmp_path: Path) -> None:
    fallback = tmp_path / "data"
    fallback.mkdir(parents=True, exist_ok=True)

    write_csv(fallback / "PoPCites_.csv", "Cites,Authors,Title,Year,Source\n10,A,T1,2020,S\n")
    write_csv(fallback / "Top3CitingPapersPerSource.csv", "Cites,Source,Title,Year\n5,S,T2,2021\n")
    write_csv(fallback / "UnifiedPapers.csv", "Cites,Authors,Title,Year,Source,Unique ID\n10,A,T1,2020,S,1\n")
    write_csv(fallback / "Edges.csv", "Citing Paper Unique ID,Cited Paper Unique ID\n1,1\n")

    data = load_data({"fallback_data_dir": str(fallback)})

    assert len(data["most_cited"]) == 1
    assert len(data["top_citing"]) == 1
    assert len(data["unified_papers"]) == 1
    assert len(data["edges"]) == 1
