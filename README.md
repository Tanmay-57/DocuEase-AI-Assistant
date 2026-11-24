# DocuEase

DocuEase is a Streamlit app for asking questions about PDF, Word, and PowerPoint files. It extracts the text, searches the relevant sections with Gemini embeddings, and answers with Gemini. Replies can be written in several languages and read aloud.

## Features

- Chat with one or more PDF, DOCX, and PPTX files
- Answers grounded in the uploaded text, with the file and page or slide shown
- Brief and detailed summaries, including long documents split into sections
- English, Spanish, French, German, Simplified Chinese, and Hindi
- Optional spoken replies
- A private in-memory index for each browser session

## Run locally

Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Add a [Google AI Studio](https://aistudio.google.com/apikey) key in `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your-key"
```

A `.env` file works as well:

```text
GOOGLE_API_KEY=your-key
```

Copy [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) if you want the optional model names too. Start the app:

```powershell
streamlit run app.py
```

Open http://localhost:8501. Upload files, choose **Process documents**, then ask a question.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub. Do not commit the API key.
2. Open [share.streamlit.io](https://share.streamlit.io) and create an app from `Tanmay-57/DocuEase-AI-Assistant`, branch `main`, main file `app.py`.
3. In Advanced settings, paste:

```toml
GOOGLE_API_KEY = "your-key"
```

4. Deploy. Set the app to private if other people should not spend your Gemini quota.

Optional secret overrides:

```toml
# Optional. Answers use gemini-3.5-flash-lite unless this is set.
# GEMINI_MODEL = "gemini-3.5-flash-lite"
# EMBED_MODEL = "gemini-embedding-2"
```

A public app uses your key for every visitor.

## Notes

- Scanned PDFs and pictures of text are not read.
- Word tables, PowerPoint tables, and speaker notes are included.
- Refreshing the page clears the uploaded files and the conversation.
- Very large uploads index the first 400 sections and say so in the sidebar.

## Project layout

- `app.py` — the Streamlit application
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — theme and the 50 MB upload limit
- `.streamlit/secrets.toml.example` — secret names, with no real key
