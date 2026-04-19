"""Streamlit MVP frontend for CSV Analyst Assistant."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from pydantic import ValidationError

# Ensure repo root is importable when running `streamlit run app/main.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.analytics import execute_structured_query
from app.llm_router import parse_question_to_structured_query

SUPPORTED_DATASETS: list[str] = [
    "online_retail_ii",
    "yellow_tripdata_2026_01",
    "insurance",
]

EXAMPLE_QUESTIONS: list[str] = [
    "What is the average charges?",
    "What is the total quantity by country?",
    "What is the average fare_amount where trip_distance > 5?",
    "What is the weekly total fare_amount over time?",
]


def _init_state() -> None:
    """Initialize session state keys used by the MVP UI."""
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    if "last_structured_query" not in st.session_state:
        st.session_state["last_structured_query"] = None
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None


def _render_examples() -> None:
    """Render clickable example questions and update input text."""
    st.markdown("**Example questions**")
    cols = st.columns(2)
    for idx, question in enumerate(EXAMPLE_QUESTIONS):
        col = cols[idx % 2]
        if col.button(question, key=f"example_{idx}", use_container_width=True):
            st.session_state["question_input"] = question


def _load_uploaded_csv() -> pd.DataFrame | None:
    """Load selected dataset from processed path, then raw path."""
    dataset_name = st.selectbox("Choose a dataset", SUPPORTED_DATASETS)
    st.session_state["selected_dataset"] = dataset_name

    processed_cleaned_path = REPO_ROOT / "data" / "processed" / f"{dataset_name}_cleaned.csv"
    processed_path = REPO_ROOT / "data" / "processed" / f"{dataset_name}.csv"
    raw_path = REPO_ROOT / "data" / "raw" / f"{dataset_name}.csv"
    candidates = [processed_path, processed_cleaned_path, raw_path]

    chosen_path = next((path for path in candidates if path.exists()), None)
    if chosen_path is None:
        _render_error(f"Dataset file not found for '{dataset_name}'.")
        return None

    try:
        st.caption(f"Source: `{chosen_path.relative_to(REPO_ROOT)}`")
        return pd.read_csv(chosen_path, low_memory=False)
    except Exception:
        _render_error("Could not read the selected dataset file.")
        return None


def _format_router_error(exc: Exception) -> str:
    """Map router/parsing failures to concise user-facing messages."""
    if isinstance(exc, requests.exceptions.RequestException):
        return "Could not reach the language model service. Please check your network and try again."
    if isinstance(exc, ValidationError):
        return "The interpreted query did not pass validation. Please rephrase your question."

    text = str(exc).strip()
    lowered = text.lower()
    if "failed to validate structuredquery" in lowered:
        return "The interpreted query did not pass validation. Please rephrase your question."
    if "api_key" in lowered or "openai_api_key" in lowered:
        return "Router configuration is missing. Please set the API key and try again."
    if "timeout" in lowered:
        return "The language model request timed out. Please try again."
    if "401" in lowered or "403" in lowered:
        return "Authentication with the language model service failed."
    if "429" in lowered:
        return "The language model service is rate-limited right now. Please retry shortly."
    return "Could not interpret your question. Please try rephrasing it."


def _format_execution_error(exc: Exception) -> str:
    """Map execution failures to concise user-facing messages."""
    if isinstance(exc, ValidationError):
        return "Query execution input was invalid. Please rephrase your question."
    return "Could not execute the query on this dataset. Please adjust your question and try again."


def _render_dataset_stats(df: pd.DataFrame) -> None:
    """Render basic dataset stats and detected columns."""
    st.markdown("**Dataset stats**")
    shape_col, count_col = st.columns(2)
    with shape_col:
        st.metric("Rows", f"{df.shape[0]}")
    with count_col:
        st.metric("Columns", f"{df.shape[1]}")

    columns_df = pd.DataFrame(
        {"column": df.columns.astype(str), "dtype": [str(dtype) for dtype in df.dtypes]}
    )
    st.markdown("**Detected columns**")
    st.dataframe(columns_df, use_container_width=True, hide_index=True)


def _render_uploaded_dataframe_preview(df: pd.DataFrame) -> None:
    """Render a small preview of the uploaded dataframe."""
    st.markdown("**Data preview**")
    st.dataframe(df.head(10), use_container_width=True)


def _render_structured_query_panel(structured_query: Any) -> None:
    """Render StructuredQuery in a readable and scan-friendly format."""
    st.subheader("Interpreted StructuredQuery")
    payload = structured_query.model_dump()
    ordered_keys = [
        "task_type",
        "metric_column",
        "aggregation",
        "groupby_column",
        "filters",
        "time_column",
        "time_granularity",
        "chart_type",
        "chart_title",
    ]
    ordered_payload = {key: payload.get(key) for key in ordered_keys}
    st.dataframe(
        pd.DataFrame(
            [{"field": key, "value": repr(value)} for key, value in ordered_payload.items()]
        ),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Raw JSON"):
        st.json(ordered_payload)


def _render_results(result: Any, structured_query: Any) -> None:
    """Render normalized backend result in a stable visual order."""
    st.subheader("Result")

    # Keep sections in deterministic order for readability.
    if result.message:
        st.info(result.message)

    if result.error:
        st.error(f"Backend error: {result.error}")
        return

    if result.value is not None:
        st.markdown("**Scalar value**")
        st.code(str(result.value))

    if result.result_type in {"table", "timeseries"} and not result.table:
        st.info("No table was returned for this query.")

    if result.table:
        st.markdown("**Result table**")
        st.dataframe(pd.DataFrame(result.table), use_container_width=True)

    expected_chart = structured_query.chart_type not in {"none", "table"}
    if expected_chart and not result.chart_data:
        st.info("No chart was returned for this query.")

    if result.chart_data:
        st.markdown("**Chart**")
        figure = result.chart_data.get("figure")
        if figure is not None:
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)
        else:
            st.info("Chart output is missing from the backend response.")


def _render_error(message: str) -> None:
    """Render a concise user-facing error message."""
    st.error(message)


def main() -> None:
    """Run the Streamlit MVP app."""
    st.set_page_config(page_title="CSV Analyst Assistant", layout="wide")
    _init_state()

    st.title("CSV Analyst Assistant")
    st.caption("Select a dataset and ask a question in natural language.")
    st.divider()

    st.header("1) Select Dataset")
    df = _load_uploaded_csv()
    if df is None:
        st.info("Select a dataset to begin.")
        return
    if df.empty:
        st.warning("The selected dataset is empty.")
        return

    st.subheader("Dataset Overview")
    preview_col, stats_col = st.columns([2, 1])
    with preview_col:
        _render_uploaded_dataframe_preview(df)
    with stats_col:
        _render_dataset_stats(df)
    st.divider()

    st.header("2) Ask a Question")
    _render_examples()
    question = st.text_input(
        "Ask a question about your data",
        key="question_input",
        placeholder="Example: What is the average value of sales?",
    )

    if st.button("Run query", type="primary"):
        cleaned_question = question.strip()
        if not cleaned_question:
            _render_error("Please enter a question before running the query.")
            return

        with st.spinner("Running query..."):
            try:
                structured_query = parse_question_to_structured_query(
                    question=cleaned_question,
                    df=df,
                )
            except Exception as exc:
                st.session_state["last_structured_query"] = None
                st.session_state["last_result"] = None
                st.session_state["last_error"] = _format_router_error(exc)
                return

            try:
                result = execute_structured_query(df=df, query=structured_query)
            except Exception as exc:
                st.session_state["last_structured_query"] = None
                st.session_state["last_result"] = None
                st.session_state["last_error"] = _format_execution_error(exc)
                return

            st.session_state["last_structured_query"] = structured_query
            st.session_state["last_result"] = result
            st.session_state["last_error"] = None

    st.divider()
    st.header("3) Output")

    if st.session_state["last_error"]:
        _render_error(f"Query failed: {st.session_state['last_error']}")
        return

    if st.session_state["last_structured_query"] is None or st.session_state["last_result"] is None:
        st.info("Run a question to view the interpreted query and results.")
        return

    _render_structured_query_panel(st.session_state["last_structured_query"])
    st.markdown("")
    _render_results(
        st.session_state["last_result"],
        st.session_state["last_structured_query"],
    )


if __name__ == "__main__":
    main()
