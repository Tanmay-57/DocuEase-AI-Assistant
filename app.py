import html
import io
import os
import re
import time
import tomllib
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from docx import Document
from dotenv import load_dotenv
from google import genai
from google.genai import types
from gtts import gTTS
from pptx import Presentation
from pypdf import PdfReader

DEFAULT_ANSWER_MODEL = "gemini-3.5-flash-lite"
DEFAULT_SUMMARY_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768
EMBED_BATCH = 64
EMBED_WORKERS = 4
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_CHUNKS = 400
RETRIEVAL_K = 8
SUMMARY_GROUP_CHARS = 40000
SINGLE_SUMMARY_CHARS = 160000
SUMMARY_PAUSE_SECONDS = 6
HISTORY_TURNS = 6
TTS_CHAR_LIMIT = 4500

STOP_WORDS = {
    "about",
    "and",
    "are",
    "can",
    "for",
    "from",
    "how",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}

QUESTION_REQUESTS = (
    "what can i ask",
    "what questions",
    "questions can i ask",
    "questions that i can ask",
    "questions should i ask",
    "suggest questions",
    "what should i ask",
    "give me questions",
    "questions to ask",
)

INSTANT_STARTERS = [
    "What is this document about?",
    "What are the key points?",
    "What names, dates, or figures does it mention?",
    "Give a short summary.",
]

LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh-CN": "Chinese (Simplified)",
    "hi": "Hindi",
}


class DocumentError(Exception):
    """A problem the user can fix by choosing different files."""


def model_is_busy(exc):
    message = str(exc).upper()
    return any(token in message for token in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


def public_error(exc):
    message = " ".join(str(exc).split())
    message = re.sub(r"[A-Za-z0-9._\-]{24,}", "[hidden]", message)
    if "API_KEY_INVALID" in message or "API key not valid" in message:
        return "The Google API key was rejected. Check GOOGLE_API_KEY in the Streamlit Cloud secrets."
    if model_is_busy(exc):
        return "The model is busy right now. Wait a moment and try the question again."
    if len(message) > 280:
        message = message[:280] + "..."
    return message


SECRET_NAMES = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY")


def _clean_secret(value):
    text = str(value).strip().strip('"').strip("'")
    return text or None


def _flatten_secrets(data):
    flat = {}
    if not isinstance(data, Mapping):
        return flat
    for key, value in data.items():
        if isinstance(value, Mapping):
            flat.update(_flatten_secrets(value))
        else:
            cleaned = _clean_secret(value)
            if cleaned:
                flat[str(key)] = cleaned
    return flat


def _secret_file_paths():
    paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
    ]
    unique = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def load_secrets():
    """Return secret values and any safe-to-show load problems."""
    load_dotenv()
    merged = {}
    problems = []
    try:
        merged.update(_flatten_secrets(dict(st.secrets)))
    except Exception as exc:
        problems.append(public_error(exc))
    for path in _secret_file_paths():
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                merged.update(_flatten_secrets(tomllib.load(handle)))
        except Exception as exc:
            problems.append(f"{path.name}: {public_error(exc)}")
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", "GEMINI_MODEL", "EMBED_MODEL"):
        env_value = os.getenv(name)
        if env_value:
            merged[name] = _clean_secret(env_value)
    return merged, problems


def api_key_from(secrets):
    for name in SECRET_NAMES:
        if secrets.get(name):
            return secrets[name]
    return None


@st.cache_resource(show_spinner=False)
def get_client(api_key):
    return genai.Client(api_key=api_key)


def clean_text(text):
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(file):
    file.seek(0)
    reader = PdfReader(file)
    passages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text())
        if text:
            passages.append(
                {"source": file.name, "location": f"Page {number}", "text": text}
            )
    return passages


def extract_docx(file):
    file.seek(0)
    document = Document(file)
    lines = [clean_text(paragraph.text) for paragraph in document.paragraphs]
    lines = [line for line in lines if line]
    for table in document.tables:
        for row in table.rows:
            cells = []
            for cell in row.cells:
                value = clean_text(cell.text)
                if value and (not cells or cells[-1] != value):
                    cells.append(value)
            if cells:
                lines.append(" | ".join(cells))
    text = clean_text("\n".join(lines))
    if not text:
        return []
    return [{"source": file.name, "location": "Document", "text": text}]


def extract_pptx(file):
    file.seek(0)
    presentation = Presentation(file)
    passages = []
    for number, slide in enumerate(presentation.slides, start=1):
        lines = []
        for shape in slide.shapes:
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = []
                    for cell in row.cells:
                        value = clean_text(cell.text)
                        if value and (not cells or cells[-1] != value):
                            cells.append(value)
                    if cells:
                        lines.append(" | ".join(cells))
            elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                value = clean_text(shape.text)
                if value:
                    lines.append(value)
        try:
            if slide.has_notes_slide:
                notes = clean_text(slide.notes_slide.notes_text_frame.text)
                if notes and not notes.isdigit():
                    lines.append(f"Notes: {notes}")
        except Exception:
            pass
        text = clean_text("\n".join(lines))
        if text:
            passages.append(
                {"source": file.name, "location": f"Slide {number}", "text": text}
            )
    return passages


def extract_files(files):
    extractors = {
        ".pdf": extract_pdf,
        ".docx": extract_docx,
        ".pptx": extract_pptx,
    }
    passages = []
    skipped = []
    unsupported = []
    for file in files:
        suffix = os.path.splitext(file.name)[1].lower()
        extractor = extractors.get(suffix)
        if extractor is None:
            unsupported.append(file.name)
            continue
        found = extractor(file)
        if found:
            passages.extend(found)
        else:
            skipped.append(file.name)
    return passages, skipped, unsupported


def chunk_passages(passages):
    chunks = []
    for passage in passages:
        text = passage["text"]
        if len(text) <= CHUNK_SIZE:
            chunks.append(dict(passage))
            continue
        start = 0
        part = 1
        while start < len(text):
            piece = text[start : start + CHUNK_SIZE].strip()
            if piece:
                chunks.append(
                    {
                        "source": passage["source"],
                        "location": f"{passage['location']}, part {part}",
                        "text": piece,
                    }
                )
                part += 1
            if start + CHUNK_SIZE >= len(text):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def content_for(text):
    return types.Content(parts=[types.Part.from_text(text=text)])


def embed_batch(client, embed_model, batch):
    result = client.models.embed_content(
        model=embed_model,
        contents=[content_for(text) for text in batch],
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    embeddings = result.embeddings or []
    if len(embeddings) != len(batch):
        raise RuntimeError(
            "The embedding service returned an unexpected number of vectors."
        )
    vectors = []
    for embedding in embeddings:
        values = getattr(embedding, "values", None)
        if not values:
            raise RuntimeError("The embedding service returned an empty vector.")
        vector = np.asarray(values, dtype="float32")
        if vector.shape[0] > EMBED_DIM:
            vector = vector[:EMBED_DIM]
        vectors.append(vector)
    return np.vstack(vectors).astype("float32")


def embed_texts(client, texts, embed_model, on_progress=None):
    if not texts:
        raise RuntimeError("There is no text to index.")
    batches = [
        texts[offset : offset + EMBED_BATCH]
        for offset in range(0, len(texts), EMBED_BATCH)
    ]
    ordered = [None] * len(batches)
    completed = 0
    workers = min(EMBED_WORKERS, len(batches))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(embed_batch, client, embed_model, batch): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            index = futures[future]
            ordered[index] = future.result()
            completed += len(batches[index])
            if on_progress:
                on_progress(completed, len(texts))
    matrix = np.vstack(ordered).astype("float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def build_index(vectors):
    matrix = np.ascontiguousarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def keywords(text):
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in STOP_WORDS
    }


def asks_for_questions(question):
    text = question.lower()
    return any(phrase in text for phrase in QUESTION_REQUESTS)


def select_sources(chunks, hits, question, limit=RETRIEVAL_K):
    semantic = [
        (chunk_id, score)
        for chunk_id, score in hits
        if 0 <= chunk_id < len(chunks)
    ]
    chosen = []
    seen = set()
    for chunk_id, score in semantic[:5]:
        chosen.append(chunk_id)
        seen.add(chunk_id)
    wanted = keywords(question)
    lexical = []
    for index, chunk in enumerate(chunks):
        if index in seen:
            continue
        overlap = wanted & keywords(f"{chunk['source']} {chunk['text']}")
        if overlap:
            lexical.append((len(overlap), index))
    lexical.sort(key=lambda item: (-item[0], item[1]))
    for _count, index in lexical:
        if len(chosen) >= limit:
            break
        chosen.append(index)
        seen.add(index)
    for chunk_id, _score in semantic:
        if len(chosen) >= limit:
            break
        if chunk_id not in seen:
            chosen.append(chunk_id)
            seen.add(chunk_id)
    scores = dict(semantic)
    sources = []
    for chunk_id in chosen:
        source = dict(chunks[chunk_id])
        source["score"] = scores.get(chunk_id)
        sources.append(source)
    return sources


def search_index(index, query_vector, k):
    scores, ids = index.search(query_vector, k)
    found = []
    for score, chunk_id in zip(scores[0], ids[0]):
        if int(chunk_id) >= 0:
            found.append((int(chunk_id), float(score)))
    return found


def _generate_once(client, chat_model, prompt, thinking_level):
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
    )
    try:
        response = client.models.generate_content(
            model=chat_model, contents=prompt, config=config
        )
    except Exception as exc:
        if "thinking" not in str(exc).lower():
            raise
        response = client.models.generate_content(model=chat_model, contents=prompt)
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("The model returned an empty response.")
    return text


def generate_text(client, chat_model, prompt, thinking_level=types.ThinkingLevel.MINIMAL):
    delay = 4
    last_error = None
    for attempt in range(3):
        try:
            return _generate_once(client, chat_model, prompt, thinking_level)
        except Exception as exc:
            last_error = exc
            if not model_is_busy(exc) or attempt == 2:
                break
            time.sleep(delay)
            delay *= 2
    if chat_model != DEFAULT_SUMMARY_MODEL and model_is_busy(last_error):
        return _generate_once(client, DEFAULT_SUMMARY_MODEL, prompt, thinking_level)
    raise last_error


def labeled_text(chunks, limit=None):
    parts = []
    used = 0
    for chunk in chunks:
        block = f"[{chunk['source']} — {chunk['location']}]\n{chunk['text']}"
        if limit is not None and used + len(block) > limit and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def group_chunks(chunks, limit):
    groups = []
    current = []
    size = 0
    for chunk in chunks:
        extra = len(chunk["text"]) + 1
        if current and size + extra > limit:
            groups.append(current)
            current = []
            size = 0
        current.append(chunk)
        size += extra
    if current:
        groups.append(current)
    return groups


def summary_instruction(length):
    if length == "Detailed":
        return (
            "Write a detailed summary that keeps the important facts, figures, "
            "and structure. Use short paragraphs or bullets."
        )
    return "Write a concise summary in one short paragraph."


def summarize_group(client, chat_model, group, language):
    prompt = (
        "Summarize this part of a longer document. Keep names, dates, and figures. "
        f"Write in {language}.\n\n{labeled_text(group)}"
    )
    return generate_text(client, chat_model, prompt)


def summarize_documents(client, chat_model, chunks, language, length, on_progress=None):
    instruction = summary_instruction(length)
    full = labeled_text(chunks)
    if len(full) <= SINGLE_SUMMARY_CHARS:
        if on_progress:
            on_progress(1, 1)
        prompt = (
            f"{instruction} Write in {language}. "
            "Use only the document text below.\n\n"
            f"{full}"
        )
        return generate_text(client, chat_model, prompt)

    groups = group_chunks(chunks, SUMMARY_GROUP_CHARS)
    partials = []
    total = len(groups) + 1
    for index, group in enumerate(groups, start=1):
        if on_progress:
            on_progress(index, total)
        partials.append(summarize_group(client, chat_model, group, language))
        if index < len(groups):
            time.sleep(SUMMARY_PAUSE_SECONDS)
    if on_progress:
        on_progress(total, total)
    combined = "\n\n".join(
        f"Part {number}:\n{text}" for number, text in enumerate(partials, start=1)
    )
    prompt = (
        f"Combine these section summaries into one summary. {instruction} "
        f"Write in {language}. Do not add facts that are not in the section summaries.\n\n"
        f"{combined}"
    )
    return generate_text(client, chat_model, prompt)


def history_text(messages):
    lines = []
    for message in messages[-HISTORY_TURNS:]:
        speaker = "User" if message["role"] == "user" else "Assistant"
        content = message["content"].strip()
        if len(content) > 800:
            content = content[:800] + "..."
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def excerpts_text(sources):
    return "\n\n".join(
        f"Source: {source['source']} ({source['location']})\n{source['text']}"
        for source in sources
    )


def answer_prompt(question, language, excerpts, prior):
    if asks_for_questions(question):
        return (
            "The reader wants questions they can ask about the uploaded documents.\n"
            f"Using only the excerpts, write 4 specific questions in {language}.\n"
            "Put one question on each line. Do not answer the questions.\n\n"
            f"Excerpts:\n{excerpts}"
        )
    return (
        "Answer the question using the excerpts.\n"
        f"Write the answer in {language}.\n"
        "Connect the wording of the question to related terms in the excerpts.\n"
        "If the excerpts contain the relevant explanation, state it directly.\n"
        "If they contain only part of the answer, give that part.\n"
        "Say that the answer is not available in the documents only when the excerpts do not discuss the topic.\n"
        "Do not add facts that the excerpts do not support.\n\n"
        f"Conversation so far:\n{prior or '(none)'}\n\n"
        f"Excerpts:\n{excerpts}\n\n"
        f"Question: {question}"
    )


def answer_question(client, chat_model, embed_model, index, chunks, question, language, messages):
    query = embed_texts(
        client,
        [f"task: search result | query: {question}"],
        embed_model,
    )
    hits = search_index(index, query, min(RETRIEVAL_K, len(chunks)))
    sources = select_sources(chunks, hits, question)
    prompt = answer_prompt(
        question,
        language,
        excerpts_text(sources),
        history_text(messages),
    )
    return (
        generate_text(
            client,
            chat_model,
            prompt,
            thinking_level=types.ThinkingLevel.LOW,
        ),
        sources,
    )


def synthesize_speech(text, lang):
    spoken = text.strip()
    truncated = len(spoken) > TTS_CHAR_LIMIT
    if truncated:
        spoken = spoken[:TTS_CHAR_LIMIT]
    audio = io.BytesIO()
    gTTS(text=spoken, lang=lang).write_to_fp(audio)
    return audio.getvalue(), truncated


def init_state():
    defaults = {
        "messages": [],
        "index": None,
        "chunks": [],
        "skipped": [],
        "truncated": False,
        "brief": "",
        "starters": [],
        "notes": [],
        "uploader_key": 0,
        "notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_documents():
    st.session_state.messages = []
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.skipped = []
    st.session_state.truncated = False
    st.session_state.brief = ""
    st.session_state.starters = []
    st.session_state.notes = []
    st.session_state.uploader_key += 1


def index_documents(client, embed_model, files, on_progress=None):
    passages, skipped, unsupported = extract_files(files)
    if unsupported:
        names = ", ".join(unsupported)
        raise DocumentError(f"These files are not supported: {names}")
    if not passages:
        raise DocumentError(
            "No readable text was found. Scanned pages and images are not supported yet."
        )
    chunks = chunk_passages(passages)
    truncated = len(chunks) > MAX_CHUNKS
    if truncated:
        chunks = chunks[:MAX_CHUNKS]
    embed_inputs = [
        f"title: {chunk['source']} | text: {chunk['text']}" for chunk in chunks
    ]
    vectors = embed_texts(client, embed_inputs, embed_model, on_progress=on_progress)
    index = build_index(vectors)
    notes = []
    if skipped:
        notes.append(
            "No text found in "
            + ", ".join(skipped)
            + ". Scanned pages and images are not supported yet."
        )
    if truncated:
        notes.append(
            f"Only the first {MAX_CHUNKS} sections were indexed so this session stays responsive."
        )
    return {
        "index": index,
        "chunks": chunks,
        "skipped": skipped,
        "truncated": truncated,
        "brief": "",
        "starters": list(INSTANT_STARTERS),
        "notes": notes,
    }


def store_index(result):
    st.session_state.index = result["index"]
    st.session_state.chunks = result["chunks"]
    st.session_state.skipped = result["skipped"]
    st.session_state.truncated = result["truncated"]
    st.session_state.brief = result["brief"]
    st.session_state.starters = result["starters"]
    st.session_state.notes = result["notes"]
    st.session_state.messages = []


def append_exchange(question, answer, sources, lang, kind="answer"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "lang": lang,
            "kind": kind,
        }
    )


APP_CSS = """
<style>
[data-testid="stHeader"] { background: transparent; }
[data-testid="stExpandSidebarButton"] {
  visibility: visible;
  background: #ffffff;
  border: 1px solid #e6e6e6;
  border-radius: 10px;
  color: #1f1f1f;
}
footer, #MainMenu { visibility: hidden; }
[data-testid="stAppViewContainer"], .stApp { background: #ffffff; }
section[data-testid="stSidebar"] {
  background: #f7f7f8;
  border-right: 1px solid #ececef;
}
section[data-testid="stSidebar"] > div { padding-top: 1.1rem; }
.block-container {
  max-width: 760px;
  padding-top: 0.6rem;
  padding-bottom: 6.5rem;
}
.de-hero { text-align: center; padding: 7vh 0 1.25rem; }
.de-mark {
  width: 52px; height: 52px; margin: 0 auto 0.9rem; border-radius: 16px;
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-weight: 650; font-size: 1.3rem;
  background: linear-gradient(135deg, #1a73e8, #7b4dff);
}
.de-hero h1 {
  margin: 0; font-size: 2.35rem; font-weight: 560; letter-spacing: -0.035em; line-height: 1.15;
  background: linear-gradient(90deg, #1a73e8 0%, #7b4dff 58%, #d946ef 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.de-hero p { margin: 0.65rem auto 0; max-width: 34rem; color: #6b6b6b; font-size: 1.02rem; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem; margin-top: 1.4rem; }
.steps div {
  text-align: left; border: 1px solid #ececef; border-radius: 16px;
  padding: 0.85rem 0.95rem; background: #fff;
}
.steps strong { display: block; margin-bottom: 0.25rem; font-size: 0.92rem; }
.steps span { color: #6b6b6b; font-size: 0.86rem; line-height: 1.35; }
.user-bubble {
  background: #f4f4f5; color: #1f1f1f; border-radius: 20px;
  padding: 0.72rem 1rem; margin-left: auto; width: fit-content; max-width: 100%;
  white-space: pre-wrap; line-height: 1.45;
}
.assistant-avatar {
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-size: 0.9rem;
  background: linear-gradient(135deg, #1a73e8, #7b4dff);
}
[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.assistant-avatar) > div:first-child {
  flex: 0 0 40px !important;
  min-width: 40px;
  max-width: 40px;
}
.doc-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: center; margin: 0.15rem 0 0.4rem; }
.doc-pills span {
  border: 1px solid #ececef; background: #fff; border-radius: 999px;
  padding: 0.22rem 0.7rem; font-size: 0.8rem; color: #444;
}
.side-brand { font-size: 1.15rem; font-weight: 650; letter-spacing: -0.02em; margin: 0; }
[data-testid="stFileUploader"] section { border-radius: 14px; background: #fff; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stMain"] [data-testid="stBaseButton-secondary"] {
  border-radius: 16px; border: 1px solid #ececef; background: #fff;
  text-align: left; min-height: 4.1rem; padding: 0.7rem 0.9rem;
  white-space: normal; line-height: 1.35;
}
[data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover {
  background: #f7f7f8; border-color: #d5d5dc;
}
[data-testid="stChatInput"] { max-width: 760px; }
[data-testid="stChatInput"] > div {
  border-radius: 28px; border: 1px solid #e3e3e8; background: #fff;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
}
@media (max-width: 700px) {
  .steps { grid-template-columns: 1fr; }
  .de-hero h1 { font-size: 1.8rem; }
  .de-hero { padding-top: 2.2rem; }
}
</style>
"""


def inject_styles():
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_sources(sources):
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            st.markdown(f"**{source['source']}** · {source['location']}")
            excerpt = source["text"].strip()
            if len(excerpt) > 500:
                excerpt = excerpt[:500] + "..."
            st.caption(excerpt)


def render_messages():
    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            _, bubble = st.columns([0.18, 0.82])
            safe = html.escape(message["content"])
            bubble.markdown(
                f'<div class="user-bubble">{safe}</div>',
                unsafe_allow_html=True,
            )
            continue
        avatar, body = st.columns([0.08, 0.92], vertical_alignment="top")
        avatar.markdown('<div class="assistant-avatar">✦</div>', unsafe_allow_html=True)
        with body:
            if message.get("kind") == "summary":
                st.caption("Summary")
            st.markdown(message["content"])
            render_sources(message.get("sources") or [])
            if st.button(
                "Listen",
                key=f"listen-{index}",
                icon=":material/volume_up:",
                type="tertiary",
            ):
                try:
                    audio, truncated = synthesize_speech(
                        message["content"], message.get("lang", "en")
                    )
                    message["audio"] = audio
                    message["audio_truncated"] = truncated
                except Exception as exc:
                    st.error(f"Could not play that reply. {public_error(exc)}")
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")
                if message.get("audio_truncated"):
                    st.caption("The recording covers the start of this reply.")


def render_doc_pills():
    counts = Counter(chunk["source"] for chunk in st.session_state.chunks)
    if not counts:
        return
    chips = "".join(f"<span>{html.escape(name)}</span>" for name in counts)
    st.markdown(f'<div class="doc-pills">{chips}</div>', unsafe_allow_html=True)


def render_hero(indexed):
    if indexed:
        subtitle = "Ask a question, or start from one of the suggestions."
    else:
        subtitle = "Add a PDF, Word, or PowerPoint file, then start the conversation."
    st.markdown(
        f"""
        <div class="de-hero">
          <div class="de-mark">D</div>
          <h1>What do you want to know?</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not indexed:
        st.markdown(
            """
            <div class="steps">
              <div><strong>1. Add files</strong><span>Use the sidebar for PDF, Word, or PowerPoint.</span></div>
              <div><strong>2. Process</strong><span>Build a private index for this session.</span></div>
              <div><strong>3. Ask</strong><span>Chat, summarize, or listen to a reply.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def sidebar(api_key, client, answer_model, summary_model, embed_model, secret_problems, secret_names):
    st.markdown('<p class="side-brand">DocuEase</p>', unsafe_allow_html=True)
    st.caption("Chat with your documents")
    if st.button("New chat", icon=":material/edit:", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Documents")
    uploads = st.file_uploader(
        "Upload PDF, DOCX, or PPTX files",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        key=f"uploads-{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    with st.expander("Reply settings"):
        language_code = st.selectbox(
            "Answer language",
            options=list(LANGUAGES),
            format_func=lambda code: LANGUAGES[code],
        )
        summary_length = st.radio(
            "Summary length",
            options=["Brief", "Detailed"],
            horizontal=True,
        )
    ready = bool(api_key and uploads)
    if st.button(
        "Process documents",
        type="primary",
        disabled=not ready,
        width="stretch",
    ):
        progress = st.progress(0.0, text="Indexing sections...")
        try:

            def on_progress(done, total):
                progress.progress(
                    done / total, text=f"Indexed {done} of {total} sections"
                )

            with st.spinner("Reading and indexing your files..."):
                result = index_documents(
                    client,
                    embed_model,
                    uploads,
                    on_progress=on_progress,
                )
            store_index(result)
            st.session_state.notice = "Documents are ready."
            st.rerun()
        except DocumentError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not process those files. {public_error(exc)}")
        finally:
            progress.empty()

    documents_ready = st.session_state.index is not None
    if not api_key:
        st.warning("Processing stays off because this server has no GOOGLE_API_KEY.")
        if secret_problems:
            for problem in secret_problems:
                st.caption(problem)
        elif secret_names:
            st.caption("Saved secret names: " + ", ".join(secret_names))
        else:
            st.caption("No secrets are saved for this deployed app.")
        st.caption("In Streamlit Cloud, open Settings, then Secrets, and save this. The local secrets file is not deployed.")
        st.code('GOOGLE_API_KEY = "your-key"', language="toml")
    elif not uploads:
        st.caption("Upload a file to enable processing.")
    elif not documents_ready:
        st.caption("Summarize turns on after processing finishes.")
    if st.button(
        "Summarize",
        disabled=not (api_key and documents_ready),
        width="stretch",
    ):
        progress = st.empty()
        try:

            def on_progress(done, total):
                if total == 1:
                    progress.caption("Writing the summary...")
                else:
                    progress.caption(f"Summarizing part {done} of {total}...")

            summary = summarize_documents(
                client,
                summary_model,
                st.session_state.chunks,
                LANGUAGES[language_code],
                summary_length,
                on_progress=on_progress,
            )
            append_exchange(
                f"Summarize the documents ({summary_length.lower()}).",
                summary,
                [],
                language_code,
                kind="summary",
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not summarize the documents. {public_error(exc)}")
        finally:
            progress.empty()

    if documents_ready:
        st.caption("In this session")
        counts = Counter(chunk["source"] for chunk in st.session_state.chunks)
        for name, count in counts.items():
            label = "section" if count == 1 else "sections"
            st.caption(f"{name} · {count} {label}")
        for note in st.session_state.notes:
            st.warning(note)

    if st.button("Remove documents", type="tertiary", width="stretch"):
        reset_documents()
        st.rerun()

    st.caption(f"Answers use {answer_model}.")
    return language_code


def ask(client, chat_model, embed_model, question, language_code):
    question = question.strip()
    if not question:
        return
    if st.session_state.index is None:
        st.warning("Process documents before asking a question.")
        return
    try:
        with st.spinner("Looking through the documents..."):
            answer, sources = answer_question(
                client,
                chat_model,
                embed_model,
                st.session_state.index,
                st.session_state.chunks,
                question,
                LANGUAGES[language_code],
                st.session_state.messages,
            )
        append_exchange(question, answer, sources, language_code)
        st.rerun()
    except Exception as exc:
        st.error(f'Could not answer "{question}". {public_error(exc)}')


def main():
    st.set_page_config(
        page_title="DocuEase",
        page_icon=":material/auto_awesome:",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    init_state()
    secrets, secret_problems = load_secrets()
    api_key = api_key_from(secrets)
    answer_model = secrets.get("GEMINI_MODEL") or DEFAULT_ANSWER_MODEL
    summary_model = DEFAULT_SUMMARY_MODEL
    embed_model = secrets.get("EMBED_MODEL") or DEFAULT_EMBED_MODEL
    client = get_client(api_key) if api_key else None

    with st.sidebar:
        language_code = sidebar(
            api_key,
            client,
            answer_model,
            summary_model,
            embed_model,
            secret_problems,
            sorted(secrets),
        )

    notice = st.session_state.notice
    if notice:
        st.session_state.notice = None
        st.toast(notice)

    if not api_key:
        st.warning(
            "This app cannot see GOOGLE_API_KEY. "
            "In Streamlit Cloud, open Settings, then Secrets, and save "
            '`GOOGLE_API_KEY = "your-key"`. Reboot the app after saving.'
        )
        render_hero(False)
        return

    indexed = st.session_state.index is not None
    if indexed:
        render_doc_pills()
    if not st.session_state.messages:
        render_hero(indexed)

    render_messages()

    if indexed and not st.session_state.messages and st.session_state.starters:
        left, right = st.columns(2)
        for number, question in enumerate(st.session_state.starters):
            column = left if number % 2 == 0 else right
            if column.button(question, key=f"starter-{number}", width="stretch", wrap=True):
                ask(client, answer_model, embed_model, question, language_code)

    prompt = st.chat_input(
        "Ask about your documents" if indexed else "Process a document to start chatting",
        disabled=not indexed,
    )
    if prompt:
        ask(client, answer_model, embed_model, prompt, language_code)


if __name__ == "__main__":
    main()
