from __future__ import annotations

from pathlib import Path
import tomllib

import pandas as pd
import streamlit as st

from data_loader import EMPTY_DATA, load_data
from graphs import (
    build_graph_citation_network,
    build_graph_most_cited,
    build_graph_top_citing,
    graph_to_plotly,
)


def load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    return tomllib.loads(text)


@st.cache_data(show_spinner=False)
def load_data_cached(config: dict) -> dict[str, pd.DataFrame]:
    return load_data(config)


def apply_filters(df: pd.DataFrame, *, min_citations: int, year_range: tuple[int, int], text_search: str) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()
    if "Cites" in filtered.columns:
        filtered = filtered[filtered["Cites"] >= min_citations]

    if "Year" in filtered.columns:
        filtered = filtered[filtered["Year"].between(year_range[0], year_range[1], inclusive="both")]

    if text_search:
        pattern = text_search.lower().strip()
        text_cols = [c for c in ("Title", "Authors", "Source") if c in filtered.columns]
        if text_cols:
            mask = filtered[text_cols].fillna("").astype(str).apply(
                lambda col: col.str.lower().str.contains(pattern, regex=False)
            )
            filtered = filtered[mask.any(axis=1)]

    return filtered


def _extract_selected_node_ids(event: object) -> list[str]:
    if event is None:
        return []

    points = []
    if isinstance(event, dict):
        points = event.get("selection", {}).get("points", [])
    else:
        selection = getattr(event, "selection", None)
        if selection is not None:
            points = getattr(selection, "points", [])

    selected = []
    for point in points:
        if isinstance(point, dict):
            customdata = point.get("customdata")
        else:
            customdata = getattr(point, "customdata", None)
        if isinstance(customdata, (list, tuple)) and customdata:
            customdata = customdata[0]
        if customdata is not None:
            selected.append(str(customdata))

    return sorted(set(selected))


def _plot_with_selection(fig, key: str, enable_selection: bool) -> list[str]:
    if not enable_selection:
        st.plotly_chart(fig, width="stretch", key=key)
        return []

    try:
        event = st.plotly_chart(
            fig,
            width="stretch",
            key=key,
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
        )
    except TypeError:
        st.plotly_chart(fig, width="stretch", key=key)
        st.info("Selection drill-down requires a newer Streamlit version with Plotly selection support.")
        return []

    return _extract_selected_node_ids(event)


def _render_exports(fig, export_df: pd.DataFrame, prefix: str) -> None:
    col1, col2, col3 = st.columns(3)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    col1.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"{prefix}.csv",
        mime="text/csv",
        key=f"{prefix}_csv",
    )

    html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
    col2.download_button(
        "Download HTML",
        data=html_bytes,
        file_name=f"{prefix}.html",
        mime="text/html",
        key=f"{prefix}_html",
    )

    png_state_key = f"{prefix}_png_bytes"
    if col3.button("Prepare PNG", key=f"{prefix}_prepare_png"):
        try:
            st.session_state[png_state_key] = fig.to_image(format="png", scale=2)
        except Exception:  # noqa: BLE001
            st.session_state[png_state_key] = None
            col3.caption("PNG export unavailable. Install `kaleido` to enable it.")

    png_bytes = st.session_state.get(png_state_key)
    if png_bytes:
        col3.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"{prefix}.png",
            mime="image/png",
            key=f"{prefix}_png",
        )


def _top_citing_details(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby("Source", dropna=True)
        .agg(top_cites=("Cites", "max"), rows_in_top3=("Title", "count"), latest_year=("Year", "max"))
        .reset_index()
    )
    grouped["node_id"] = "source::" + grouped["Source"].astype(str)
    return grouped[["node_id", "Source", "top_cites", "rows_in_top3", "latest_year"]]


def main() -> None:
    st.set_page_config(page_title="Revue Finance Citation Network", layout="wide")
    st.title("Revue Finance Citation Network")
    st.write(
        "Interactive exploration of citation data for the most cited papers in Revue Finance "
        "(2008-2023), based on Harzing PoP exports and your curated CSV files."
    )

    settings_path = Path(__file__).parent / "settings.toml"
    settings = load_settings(settings_path)
    config = settings.get("data", {})
    cloud_config = settings.get("cloud", {})
    if "fallback_data_dir" not in config:
        config["fallback_data_dir"] = cloud_config.get("fallback_data_dir", "data")

    if not config:
        st.warning("No `data` section found in settings.toml. Using empty placeholders.")

    data = load_data_cached(config)
    warnings = data.get("_warnings", [])

    with st.sidebar:
        st.header("Controls")
        if warnings:
            for warning in warnings:
                st.warning(warning)

        years = []
        for key in ("most_cited", "top_citing", "unified_papers"):
            df = data.get(key, EMPTY_DATA)
            if not df.empty and "Year" in df.columns:
                years.extend(df["Year"].dropna().astype(int).tolist())
        if years:
            min_year, max_year = min(years), max(years)
        else:
            min_year, max_year = 2008, 2023

        all_cites = []
        for key in ("most_cited", "top_citing", "unified_papers"):
            df = data.get(key, EMPTY_DATA)
            if not df.empty and "Cites" in df.columns:
                all_cites.extend(df["Cites"].dropna().astype(int).tolist())
        max_cites = max(all_cites) if all_cites else 100

        min_citations = st.slider("Min citations", min_value=0, max_value=max_cites, value=0)
        year_range = st.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
        text_search = st.text_input("Text search (title/author/source)", value="")
        node_scale = st.slider("Node size scale", min_value=0.5, max_value=4.0, value=1.0, step=0.1)
        layout_seed = st.number_input("Layout seed", min_value=1, max_value=9999, value=24, step=1)
        enable_selection = st.checkbox("Enable node selection (reruns app)", value=False)

        if "reset_counter" not in st.session_state:
            st.session_state["reset_counter"] = 0
        if st.button("Reset view"):
            st.session_state["reset_counter"] += 1

    ui_state = {
        "node_size_scale": node_scale,
        "layout_seed": int(layout_seed),
        "reset_counter": int(st.session_state["reset_counter"]),
    }

    tabs = st.tabs(["Most Cited RF Papers", "Top Citing Journals/Papers", "RF Citation Network"])

    with tabs[0]:
        df = apply_filters(
            data.get("most_cited", EMPTY_DATA),
            min_citations=min_citations,
            year_range=year_range,
            text_search=text_search,
        )
        if df.empty:
            st.info("No papers match the current filters.")
        else:
            df_details = df.copy()
            df_details["node_id"] = "paper::" + df_details["Title"].astype(str) + "::" + df_details["Year"].astype(str)

            graph = build_graph_most_cited(df)
            fig = graph_to_plotly(graph, view_mode="most_cited", ui_state=ui_state)
            selected_node_ids = _plot_with_selection(
                fig, key=f"tab1_{ui_state['reset_counter']}", enable_selection=enable_selection
            )
            _render_exports(fig, df, prefix="rf_most_cited")

            if selected_node_ids:
                selected_df = df_details[df_details["node_id"].isin(selected_node_ids)].drop(columns=["node_id"])
                st.subheader("Selected node details")
                st.dataframe(selected_df, width="stretch")
            elif enable_selection:
                st.caption("Select nodes (click or lasso) to display detailed rows.")
        st.caption(
            "Data source: Harzing PoP export of most cited Revue Finance papers. "
            "Node size and color represent citations."
        )

    with tabs[1]:
        df = apply_filters(
            data.get("top_citing", EMPTY_DATA),
            min_citations=min_citations,
            year_range=year_range,
            text_search=text_search,
        )
        if df.empty:
            st.info("No citing journals/papers match the current filters.")
        else:
            details = _top_citing_details(df)
            graph = build_graph_top_citing(df)
            fig = graph_to_plotly(graph, view_mode="top_citing", ui_state=ui_state)
            selected_node_ids = _plot_with_selection(
                fig, key=f"tab2_{ui_state['reset_counter']}", enable_selection=enable_selection
            )
            _render_exports(fig, details.drop(columns=["node_id"], errors="ignore"), prefix="rf_top_citing_sources")

            if selected_node_ids:
                selected_df = details[details["node_id"].isin(selected_node_ids)].drop(columns=["node_id"])
                st.subheader("Selected node details")
                st.dataframe(selected_df, width="stretch")
            elif enable_selection:
                st.caption("Select nodes (click or lasso) to display source-level details.")
        st.caption(
            "Data source: top 3 most cited citing papers per source journal. "
            "Nodes represent journals/sources linked to citation impact."
        )

    with tabs[2]:
        papers = apply_filters(
            data.get("unified_papers", EMPTY_DATA),
            min_citations=min_citations,
            year_range=year_range,
            text_search=text_search,
        )
        edges = data.get("edges", EMPTY_DATA)
        if papers.empty or edges.empty:
            st.info("No citation-network data available for current filters.")
        else:
            if "Unique ID" in papers.columns and not papers.empty:
                ids = set(papers["Unique ID"].astype(int).tolist())
                filtered_edges = edges[
                    edges["Citing Paper Unique ID"].isin(ids) & edges["Cited Paper Unique ID"].isin(ids)
                ]
            else:
                filtered_edges = EMPTY_DATA.copy()

            graph = build_graph_citation_network(papers, filtered_edges)
            if graph.number_of_nodes() == 0:
                st.info("No network remains after filtering.")
            else:
                fig = graph_to_plotly(graph, view_mode="citation_network", ui_state=ui_state)
                selected_node_ids = _plot_with_selection(
                    fig, key=f"tab3_{ui_state['reset_counter']}", enable_selection=enable_selection
                )

                cited_ids = set(filtered_edges["Cited Paper Unique ID"].astype(int).tolist()) if not filtered_edges.empty else set()
                citing_ids = set(filtered_edges["Citing Paper Unique ID"].astype(int).tolist()) if not filtered_edges.empty else set()

                details = papers.copy()
                details["node_id"] = details["Unique ID"].astype(str)
                details["Node Type"] = details["Unique ID"].apply(
                    lambda x: "cited" if int(x) in cited_ids else ("citing" if int(x) in citing_ids else "isolated")
                )
                details = details[details["Node Type"] != "isolated"]

                _render_exports(fig, details.drop(columns=["node_id"], errors="ignore"), prefix="rf_citation_network")

                if selected_node_ids:
                    selected_df = details[details["node_id"].isin(selected_node_ids)].drop(columns=["node_id"])
                    st.subheader("Selected node details")
                    st.dataframe(selected_df, width="stretch")
                elif enable_selection:
                    st.caption("Select nodes (click or lasso) to display paper-level details.")
        st.caption(
            "Firebrick nodes represent papers published in Revue Finance. "
            "Orange nodes represent citing papers from outside Revue Finance."
        )


if __name__ == "__main__":
    main()