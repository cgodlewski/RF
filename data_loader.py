from __future__ import annotations

from pathlib import Path

import pandas as pd

EMPTY_DATA = pd.DataFrame()

REQUIRED_COLUMNS = {
    "most_cited": {"Cites", "Authors", "Title", "Year", "Source"},
    "top_citing": {"Cites", "Source", "Title", "Year"},
    "unified_papers": {"Cites", "Authors", "Title", "Year", "Source", "Unique ID"},
    "edges": {"Citing Paper Unique ID", "Cited Paper Unique ID"},
}

DEFAULT_FILENAMES = {
    "most_cited": "PoPCites_.csv",
    "top_citing": "Top3CitingPapersPerSource.csv",
    "unified_papers": "UnifiedPapers.csv",
    "edges": "Edges.csv",
}

CONFIG_KEYS = {
    "most_cited": "most_cited_csv",
    "top_citing": "top_citing_csv",
    "unified_papers": "unified_papers_csv",
    "edges": "edges_csv",
}


def standardize_authors(name: str) -> str:
    if not isinstance(name, str):
        return ""
    normalized = name.replace("*", "").strip()
    normalized = normalized.replace("1", "").replace("2", "").replace("3", "")
    parts = normalized.split()
    return " ".join(parts)


def standardize_source(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.replace("&", "and ").replace(".", "").replace(",", "").strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: col.replace("\ufeff", "").strip() for col in df.columns}
    return df.rename(columns=renamed)


def coerce_common_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Cites" in out.columns:
        out["Cites"] = pd.to_numeric(out["Cites"], errors="coerce").fillna(0).astype(int)
    if "Year" in out.columns:
        out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
        out = out[out["Year"].notna()].copy()
        out["Year"] = out["Year"].astype(int)
    if "Unique ID" in out.columns:
        out["Unique ID"] = pd.to_numeric(out["Unique ID"], errors="coerce").astype("Int64")
        out = out[out["Unique ID"].notna()].copy()
        out["Unique ID"] = out["Unique ID"].astype(int)
    if "Citing Paper Unique ID" in out.columns:
        out["Citing Paper Unique ID"] = pd.to_numeric(out["Citing Paper Unique ID"], errors="coerce").astype("Int64")
    if "Cited Paper Unique ID" in out.columns:
        out["Cited Paper Unique ID"] = pd.to_numeric(out["Cited Paper Unique ID"], errors="coerce").astype("Int64")
    if "Citing Paper Unique ID" in out.columns and "Cited Paper Unique ID" in out.columns:
        out = out[out["Citing Paper Unique ID"].notna() & out["Cited Paper Unique ID"].notna()].copy()
        out["Citing Paper Unique ID"] = out["Citing Paper Unique ID"].astype(int)
        out["Cited Paper Unique ID"] = out["Cited Paper Unique ID"].astype(int)
    return out


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")


def _resolve_path(name: str, configured_path: str, fallback_dir: str, warnings: list[str]) -> str:
    if configured_path:
        configured = Path(configured_path)
        if configured.exists():
            return str(configured)

    fallback = Path(fallback_dir) / DEFAULT_FILENAMES[name]
    if fallback.exists():
        if configured_path:
            warnings.append(f"[{name}] Primary path unavailable; using fallback file: {fallback}")
        return str(fallback)

    if configured_path:
        warnings.append(f"[{name}] File not found: {configured_path}")
    else:
        warnings.append(f"[{name}] Missing file path in settings and no fallback at: {fallback}")
    return ""


def load_one(name: str, path: str, warnings: list[str]) -> pd.DataFrame:
    if not path:
        return EMPTY_DATA.copy()

    try:
        df = read_csv(path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="windows-1252", sep=";", on_bad_lines="skip")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"[{name}] Could not read file ({path}): {exc}")
            return EMPTY_DATA.copy()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"[{name}] Could not read file ({path}): {exc}")
        return EMPTY_DATA.copy()

    df = normalize_columns(df)

    missing = REQUIRED_COLUMNS[name] - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        warnings.append(f"[{name}] Missing required columns: {missing_cols}")
        return EMPTY_DATA.copy()

    if "Authors" in df.columns:
        df["Authors"] = df["Authors"].apply(standardize_authors)
    if "Source" in df.columns:
        df["Source"] = df["Source"].apply(standardize_source)

    return coerce_common_types(df)


def load_data(config: dict) -> dict[str, pd.DataFrame]:
    warnings: list[str] = []
    fallback_dir = config.get("fallback_data_dir", "data")

    resolved_paths = {
        name: _resolve_path(name, config.get(CONFIG_KEYS[name], ""), fallback_dir, warnings)
        for name in CONFIG_KEYS
    }

    data = {
        "most_cited": load_one("most_cited", resolved_paths["most_cited"], warnings),
        "top_citing": load_one("top_citing", resolved_paths["top_citing"], warnings),
        "unified_papers": load_one("unified_papers", resolved_paths["unified_papers"], warnings),
        "edges": load_one("edges", resolved_paths["edges"], warnings),
        "_warnings": warnings,
    }
    return data
