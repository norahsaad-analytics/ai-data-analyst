# AI Data Analyst

An AI-powered web app that analyses uploaded data and produces a clear
analytical summary. Built with Python and Streamlit.

**Live app:** https://ai-data-analyst-7jtnaqhtruykmbji5uurjm.streamlit.app/

## Features
- Upload a dataset and explore it in an interactive dashboard
- Automatic analytical summary written in plain English
- Charts and key statistics generated from the data

## Tools
Python, Streamlit, pandas, AI model API

## How to run locally
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Configuration
The app needs an API key for the AI model. Add it as a secret
(for example in `.streamlit/secrets.toml`). Never commit keys to the repository.

## Note
The sample data used is not included. Upload your own file to try the app.
