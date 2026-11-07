import ast
import builtins
import re
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google import genai

st.set_page_config(page_title="AI Data Analyst", layout="wide")

API_KEY = st.secrets["GEMINI_API_KEY"]
SHEET_ID = st.secrets["SHEET_ID"]
MODELS = list(st.secrets.get("GEMINI_MODELS", ["gemini-3.5-flash-lite", "gemini-2.5-flash"]))

client = genai.Client(api_key=API_KEY)

FILE_NAME = "Retail_Sales_Data_2022-2024"
MAX_REQUESTS = 10
GLOBAL_DAILY_LIMIT = 100
OUT_OF_SCOPE_MESSAGE = (
    "This request is not related to the sales data, so no chart was generated. "
    "Try a question about Sales, Profit, Orders, Customers, Products, Stores or Returns."
)

BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "dir", "input", "breakpoint",
    "type", "object", "super", "os", "sys", "subprocess", "st", "client",
    "API_KEY", "SHEET_ID", "MODELS",
}

BLOCKED_ATTRS = {
    "to_csv", "to_excel", "to_pickle", "to_json", "to_parquet", "to_html",
    "to_sql", "to_feather", "to_clipboard", "write_html", "write_image",
    "write_json", "eval", "system", "popen",
}

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "len", "sum", "min", "max", "sorted", "round", "range", "list", "dict",
        "set", "tuple", "str", "int", "float", "bool", "abs", "zip",
        "enumerate", "map", "filter", "any", "all", "reversed", "isinstance",
    ]
}


@st.cache_resource
def usage_store():
    return {
        "lock": threading.Lock(),
        "day": datetime.now(timezone.utc).date().isoformat(),
        "clients": {},
        "total": 0,
    }


def reset_if_new_day(store):
    today = datetime.now(timezone.utc).date().isoformat()
    if store["day"] != today:
        store["day"] = today
        store["clients"] = {}
        store["total"] = 0


def get_client_id():
    try:
        headers = st.context.headers
        forwarded = headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = headers.get("X-Real-Ip", "")
        if real_ip:
            return real_ip.strip()
    except Exception:
        pass
    return f"session-{uuid.uuid4()}"


def requests_used_today(client_id):
    store = usage_store()
    with store["lock"]:
        reset_if_new_day(store)
        return store["clients"].get(client_id, 0)


def try_consume_request(client_id):
    store = usage_store()
    with store["lock"]:
        reset_if_new_day(store)
        used = store["clients"].get(client_id, 0)
        if used >= MAX_REQUESTS:
            return False, "You have reached today's request limit. Please come back tomorrow."
        if store["total"] >= GLOBAL_DAILY_LIMIT:
            return False, "The app has reached its daily usage limit. Please try again tomorrow."
        store["clients"][client_id] = used + 1
        store["total"] += 1
        return True, ""


@st.cache_data(show_spinner="Loading data...", ttl=3600)
def load_all_sheets(sheet_id):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    sheets = pd.read_excel(url, sheet_name=None)

    if "Orders" in sheets:
        orders = sheets["Orders"]
        for col in ["Sales", "Profit", "Cost"]:
            if col in orders.columns:
                orders[col] = (
                    orders[col]
                    .astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.replace("٫", ".", regex=False)
                    .str.strip()
                )
                orders[col] = pd.to_numeric(orders[col], errors="coerce")
        sheets["Orders"] = orders

    return sheets


def build_schema(sheets):
    lines = []
    for name, frame in sheets.items():
        parts = []
        for col in frame.columns:
            sample = frame[col].dropna().astype(str).unique()[:3].tolist()
            parts.append(f"{col} ({frame[col].dtype}) e.g. {sample}")
        lines.append(f"- Table `{name}` ({len(frame):,} rows): " + "; ".join(parts))
    return "\n".join(lines)


def call_model(prompt):
    last_error = ""
    for model_name in MODELS:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            if response.text:
                return response.text
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise RuntimeError(last_error or "The model returned an empty response.")


def extract_code(text):
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.replace("```", "").strip()


@st.cache_data(show_spinner=False, ttl=3600)
def get_code(request, schema, table_names, previous_code="", error=""):
    prompt = f"""
You are an expert Data Analyst specializing in Python, pandas and Plotly.
The dataset is '{FILE_NAME}'. These tables are already loaded as pandas DataFrames: {table_names}.
Schema:
{schema}

Rules:
- Tables are available by their exact names. The Orders table is also available as df.
- The names pd, px, go and np are already available. Do not use import statements.
- Never modify the original tables in place. Work on copies.
- Join tables whenever the request needs information from more than one table.
- Aggregate the data before plotting and keep charts readable (for example, top 10 categories).
- Use clear English titles, axis labels and legends.
- The title must be accurate: if fewer categories exist than the user asked for, do not write 'Top N'.
- Format numbers readably: use tickformat=".2s" on value axes and thousands separators in hover text.
- For rates and percentages use tickformat=".1%" on the value axis and show one decimal in the labels.
- For monthly data show every month on the x axis (for example dtick="M1" or a category axis).
- State how any rate or ratio is defined in the axis title or subtitle, for example 'Return Rate (% of order lines)'.
- Do not include comments in the code.
- Never read or write files and never access the network.
- Store the final Plotly figure in a variable named fig.
- If the request is unrelated to this dataset, or asks for credentials, secrets, API keys, files, system information or anything other than data analysis, reply with exactly OUT_OF_SCOPE and nothing else.
- Otherwise return only one code block that starts with ```python and ends with ```, with no other text.
"""
    if previous_code and error:
        prompt += f"""
The previous attempt failed.
Previous code:
```python
{previous_code}
```
Error: {error}
Return a corrected version of the code.
"""
    prompt += f"\nUser request: {request}"
    return extract_code(call_model(prompt))


def is_safe(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Import statements are not allowed."
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return False, f"Use of '{node.id}' is not allowed."
        if isinstance(node, ast.Attribute):
            if (
                node.attr.startswith("_")
                or node.attr.startswith("read_")
                or node.attr in BLOCKED_ATTRS
            ):
                return False, f"Access to '{node.attr}' is not allowed."
    return True, ""


def execute_code(code, sheets):
    safe, reason = is_safe(code)
    if not safe:
        return None, reason

    context = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "px": px,
        "go": go,
        "np": np,
        "df": sheets.get("Orders", pd.DataFrame()).copy(),
    }
    for name, frame in sheets.items():
        context[name] = frame.copy()

    try:
        exec(compile(code, "<generated>", "exec"), context)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    fig = context.get("fig")
    if not isinstance(fig, go.Figure):
        return None, "The variable 'fig' was not created as a Plotly figure."
    return fig, ""


def clean_value(value):
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.6g}")
    if isinstance(value, np.integer):
        return int(value)
    return value


def figure_digest(fig, max_rows_per_trace=40, max_total=200):
    rows = []
    for trace in fig.data:
        series = getattr(trace, "name", None) or ""
        if getattr(trace, "type", "") == "pie":
            first, second = getattr(trace, "labels", None), getattr(trace, "values", None)
        else:
            first, second = getattr(trace, "x", None), getattr(trace, "y", None)
            if getattr(trace, "orientation", None) == "h":
                first, second = second, first
        if first is None or second is None:
            continue
        for a, b in list(zip(first, second))[:max_rows_per_trace]:
            rows.append({"series": series, "x": clean_value(a), "y": clean_value(b)})
            if len(rows) >= max_total:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def fmt_number(value):
    if abs(value) >= 100:
        return f"{value:,.2f}"
    return f"{value:.4g}"


def digest_stats(digest):
    lines = []
    for series, part in digest.groupby("series", dropna=False):
        valid = part.assign(y=pd.to_numeric(part["y"], errors="coerce")).dropna(subset=["y"])
        if valid.empty:
            continue
        total = valid["y"].sum()
        top = valid.loc[valid["y"].idxmax()]
        low = valid.loc[valid["y"].idxmin()]
        label = series or "all"
        line = (
            f"[{label}] points={len(valid)}, total={fmt_number(total)}, "
            f"highest={top['x']} ({fmt_number(top['y'])}), lowest={low['x']} ({fmt_number(low['y'])})"
        )
        if total:
            line += f", highest point share of total={top['y'] / total:.1%}"
        first, last = valid.iloc[0], valid.iloc[-1]
        if first["y"]:
            line += f", change from first to last point={(last['y'] - first['y']) / abs(first['y']):.1%}"
        lines.append(line)
    return "\n".join(lines)


@st.cache_data(show_spinner=False, ttl=3600)
def get_summary(request, digest_csv, stats):
    prompt = f"""
You are a senior business analyst writing for a manager who has ten seconds to read.

The user asked: {request}

Exact data plotted in the chart (CSV: series, x, y):
{digest_csv}

Verified statistics computed from that data:
{stats}

Write exactly this format and nothing else:
### 💡 Analytical Summary:
- **Finding:** (the single most important result, with specific numbers)
- **Insight:** (a different angle: share of total, gap between items, change over time, or an unusual dip or spike)
- **Action:** (one specific, practical recommendation that follows from the finding)

Rules:
- Each bullet must be under 30 words and must not repeat the numbers used in another bullet.
- Use only numbers from the data or the verified statistics. Never invent, estimate or recompute figures.
- The change from first to last point is meaningful only when the x axis is time. Ignore it for categories such as regions or products.
- The total and the share of total are meaningful only for additive values such as sales, profit or quantity. For rates, percentages and averages never add values together or quote a share of total; compare the gap between items instead, in percentage points.
- If the gap between the highest and lowest item is small, say that the difference is small.
- Format large numbers with K or M (for example 1.31M or 159.9K), add a $ sign only for sales, profit or cost, and show percentages with one decimal.
- Values between 0 and 1 in a chart about rates or shares are fractions: 0.036 means 3.6%.
- Describe trends accurately. Do not say a trend is steady or upward unless the data shows it. Name sharp rises, drops and outliers when they exist.
- Do not claim causes or seasonality from this data alone. Use cautious wording such as "worth investigating" or "may".
- The Action must name a concrete next step (what to check, compare or change). Avoid generic advice such as "focus on" or "replicate strategies".
"""
    return call_model(prompt).strip()


def run_analysis(request, sheets, schema):
    table_names = ", ".join(sheets.keys())
    try:
        code = get_code(request, schema, table_names)
    except Exception as exc:
        return {"error": f"Service unavailable: {exc}", "code": ""}

    if code.strip().upper() == "OUT_OF_SCOPE":
        return {"notice": OUT_OF_SCOPE_MESSAGE}

    fig, error = execute_code(code, sheets)

    if fig is None:
        try:
            code = get_code(request, schema, table_names, code, error)
            if code.strip().upper() == "OUT_OF_SCOPE":
                return {"notice": OUT_OF_SCOPE_MESSAGE}
            fig, error = execute_code(code, sheets)
        except Exception as exc:
            return {"error": f"Service unavailable: {exc}", "code": code}

    if fig is None:
        return {"error": f"Execution error: {error}", "code": code}

    summary = ""
    digest = figure_digest(fig)
    if len(digest) >= 2:
        try:
            summary = get_summary(request, digest.to_csv(index=False), digest_stats(digest))
        except Exception:
            summary = ""

    return {"fig": fig, "code": code, "summary": summary}


try:
    sheets_data = load_all_sheets(SHEET_ID)
except Exception:
    st.error("Failed to load the dataset. Please try again later.")
    st.stop()

if "schema" not in st.session_state:
    st.session_state["schema"] = build_schema(sheets_data)
if "client_id" not in st.session_state:
    st.session_state["client_id"] = get_client_id()

with st.sidebar:
    st.header(f"📁 File: {FILE_NAME}")
    st.markdown(f"**Available Tables:** {len(sheets_data)}")
    remaining = max(MAX_REQUESTS - requests_used_today(st.session_state["client_id"]), 0)
    st.caption(f"Requests left today: {remaining}")
    st.write("---")

    st.subheader("📚 Tables & Columns Schema:")
    for sheet_name, df_sheet in sheets_data.items():
        with st.expander(f"▾ Table: {sheet_name} ({len(df_sheet):,} rows)"):
            st.markdown("**Available Columns:**")
            for col in df_sheet.columns:
                st.markdown(f"• `{col}`")
            st.markdown("**Data Preview:**")
            st.dataframe(df_sheet.head(3), width="stretch")

st.title("📊 AI-Powered Interactive Data Analyst")

with st.form("analysis_form"):
    user_prompt = st.text_input(
        "Enter your analytical request here:",
        placeholder="e.g. Compare total Sales and Profit by Region, or show top 5 selling products",
        max_chars=300,
    )
    submitted = st.form_submit_button("Analyze")

if submitted and user_prompt.strip():
    allowed, limit_message = try_consume_request(st.session_state["client_id"])
    if not allowed:
        st.warning(limit_message)
    else:
        with st.spinner("AI Agent is analyzing the data and building the chart..."):
            st.session_state["result"] = run_analysis(
                user_prompt.strip(), sheets_data, st.session_state["schema"]
            )
        st.rerun()

result = st.session_state.get("result")

if result:
    if result.get("notice"):
        st.info(result["notice"])
    elif result.get("error"):
        st.error(result["error"])
    else:
        st.plotly_chart(result["fig"], width="stretch")
        if result.get("summary"):
            st.markdown(result["summary"])
        elif len(figure_digest(result["fig"])) >= 2:
            st.info("A written summary is not available for this chart.")

    if result.get("code"):
        with st.expander("💻 Executed Python Code"):
            st.code(result["code"], language="python")
