# AI Data Analyst

An AI-powered web app that turns plain-English questions into interactive
charts and a short analytical summary. Built with Python and Streamlit.

**Live app:** https://ai-data-analyst-7jtnaqhtruykmbji5uurjm.streamlit.app/

## How it works
1. The app loads a retail sales dataset (2022-2024) made of six tables:
   Orders, Customers, Products, Stores, Returns and Calendar.
2. You type a request, for example "Compare total Sales and Profit by Region".
3. A Gemini model writes Python/Plotly code for the request.
4. The app runs the code, shows the interactive chart, and writes a summary
   with key findings and a business recommendation.
5. The generated code is shown in an expandable section for transparency.

## Tools
Python, Streamlit, pandas, Plotly, Google Gemini API

## Data
A public retail sales dataset from Kaggle, loaded into the app from a
spreadsheet.
