import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
import re
import time

st.set_page_config(page_title="AI Data Analyst", layout="wide")

API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

FILE_NAME = "Retail_Sales_Data_2022-2024"

@st.cache_data
def load_all_sheets():
    sheet_id = "1QklkvrikSFDKY2-B8blsG9abArMY8KU6D16iWGfbGIc"
    cloud_excel_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    sheets_dict = pd.read_excel(cloud_excel_url, sheet_name=None)
    
    if "Orders" in sheets_dict:
        orders_df = sheets_dict["Orders"]
        for col in ['Sales', 'Profit', 'Cost']:
            if col in orders_df.columns:
                orders_df[col] = (
                    orders_df[col].astype(str)
                    .str.replace('$', '', regex=False)
                    .str.replace(',', '', regex=False)
                    .str.replace('٫', '.', regex=False)
                    .str.strip()
                )
                orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce')
        sheets_dict["Orders"] = orders_df

    return sheets_dict

sheets_data = load_all_sheets()

with st.sidebar:
    st.header(f"📁 File: {FILE_NAME}")
    st.markdown(f"**Available Tables:** {len(sheets_data)}")
    st.write("---")
    
    st.subheader("📚 Tables & Columns Schema:")
    for sheet_name, df_sheet in sheets_data.items():
        with st.expander(f"▾ Table: {sheet_name} ({len(df_sheet):,} rows)"):
            st.markdown("**Available Columns:**")
            for col in df_sheet.columns:
                st.markdown(f"• `{col}`")
            st.markdown("**Data Preview:**")
            st.dataframe(df_sheet.head(3), use_container_width=True)

st.title("📊 AI-Powered Interactive Data Analyst")

user_prompt = st.text_input(
    "Enter your analytical request here:",
    placeholder="e.g. Compare total Sales and Profit by Region, or show top 5 selling products"
)

if user_prompt:
    with st.spinner("AI Agent is analyzing schemas and generating the interactive chart..."):
        schema_info = ""
        for name, d in sheets_data.items():
            schema_info += f"- Table `{name}`: contains columns {d.columns.tolist()}\n"

        system_prompt = f"""
You are an expert AI Data Analyst and Data Engineer specializing in Python and Plotly.
The primary dataset name is '{FILE_NAME}'.
The dataset includes 6 tables available as pandas DataFrames in Python:
{schema_info}

Technical Guidelines:
- All tables are directly accessible by their exact names: Orders, Customers, Products, Stores, Returns, Calendar.
- The most frequently used table is Orders (also accessible as df).
- Perform merges/joins whenever the query demands relationship across tables.
- Ensure figure titles, axis labels, and legends are clearly formatted in English.

Response Format (strict):
Part 1: Pure executable Python code enclosed ONLY in ```python and ``` that builds an interactive Plotly figure using plotly.express (px) or plotly.graph_objects. Save the final figure strictly in a variable named `fig`.
Part 2: Immediately after closing the code block, provide a markdown analytical summary in English:
### 💡 Analytical Summary:
- (Key numerical finding or primary insight)
- (Pattern, trend, or comparative observation)
- (Actionable business recommendation or takeaway)
"""

        candidate_models = [
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.6-flash",
            "gemini-2.5-flash"
        ]

        response = None
        last_error = ""

        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"{system_prompt}\n\nUser request: {user_prompt}"
                )
                break
            except Exception as e:
                last_error = str(e)
                time.sleep(0.5)
                continue

        if response:
            text_output = response.text

            code_match = re.search(r"```python(.*?)```", text_output, re.DOTALL)
            if code_match:
                clean_code = code_match.group(1).strip()
                explanation = text_output.split("```")[-1].strip()
            else:
                clean_code = text_output.replace("```", "").strip()
                explanation = ""

            execution_context = {"pd": pd, "px": px, "df": sheets_data.get("Orders", pd.DataFrame())}
            for s_name, s_df in sheets_data.items():
                execution_context[s_name] = s_df

            try:
                exec(clean_code, globals(), execution_context)

                if "fig" in execution_context:
                    st.plotly_chart(execution_context["fig"], use_container_width=True)
                else:
                    st.error("Figure variable 'fig' was not found in the generated script.")

                if explanation:
                    st.markdown(explanation)

                with st.expander("💻 Executed Python Code"):
                    st.code(clean_code, language="python")

            except Exception as run_err:
                st.error(f"Execution Error: {run_err}")
                with st.expander("Inspect Error Code"):
                    st.code(clean_code, language="python")
        else:
            st.error(f"Service Unavailable: High demand on models. Details: {last_error}")