# rag_app_fixed_polished_llm_judge_no_raw_eval.py
"""
Polished, production-friendly RAG Streamlit app for sports-event QA.
Modified to use an LLM as an automatic judge that returns precision/recall/F1
but with the following removals per user request:
 - Do NOT display the LLM raw non-JSON output anywhere in the UI.
 - Remove the optional "expected answer" manual input column.

Other behavior remains: index creation, retrieval, generation, token-overlap metrics
are kept as a fallback and shown to the user.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

# Robust imports for different langchain / langchain-community versions
LANGCHAIN_IMPORT_ERROR = None
TRANSFORMERS_IMPORT_ERROR = None

# Try to import TextLoader, Chroma, HuggingFaceEmbeddings from common places
TextLoader = None
Chroma = None
HuggingFaceEmbeddings = None
RecursiveCharacterTextSplitter = None

try:
    # prefer official langchain paths
    from langchain.document_loaders import TextLoader as _TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter as _Splitter
    from langchain.embeddings import HuggingFaceEmbeddings as _HFEmb
    from langchain.vectorstores import Chroma as _Chroma
    TextLoader = _TextLoader
    RecursiveCharacterTextSplitter = _Splitter
    HuggingFaceEmbeddings = _HFEmb
    Chroma = _Chroma
except Exception:
    try:
        # fallback to community package (older examples use this)
        from langchain_community.document_loaders import TextLoader as _TextLoader
        from langchain_community.embeddings import HuggingFaceEmbeddings as _HFEmb
        from langchain_community.vectorstores import Chroma as _Chroma
        from langchain.text_splitter import RecursiveCharacterTextSplitter as _Splitter
        TextLoader = _TextLoader
        RecursiveCharacterTextSplitter = _Splitter
        HuggingFaceEmbeddings = _HFEmb
        Chroma = _Chroma
    except Exception as e:
        # keep the exception for a helpful error message later
        LANGCHAIN_IMPORT_ERROR = e
        # leave the names as None so checks below can show helpful messages

# transformers
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
except Exception as e:
    TRANSFORMERS_IMPORT_ERROR = e


try:
    import nltk
except Exception:
    nltk = None

from dotenv import load_dotenv
load_dotenv()

# ---------------- Configuration ----------------
DEFAULT_DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", "docs"))
CHROMA_PERSIST_DIR = Path(os.getenv("RAG_CHROMA_DIR", "chroma_db"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
RAG_MODEL = os.getenv("RAG_RAG_MODEL", "google/flan-t5-base")
MAX_GENERATION_LENGTH = int(os.getenv("RAG_MAX_GEN_LEN", 512))
SPLIT_CHUNK_SIZE = int(os.getenv("RAG_SPLIT_CHUNK_SIZE", 1000))
SPLIT_CHUNK_OVERLAP = int(os.getenv("RAG_SPLIT_CHUNK_OVERLAP", 200))
DEFAULT_K = int(os.getenv("RAG_DEFAULT_K", 3))

PROMPT_TEMPLATE_COT = (
    "You are a helpful assistant. Use ONLY the following context:\n\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer strictly from the context above. "
    "Give a numbered, step-by-step explanation (4–6 steps). "
    "Do NOT add facts not present in the context. "
    "If a detail is missing, explicitly say so. "
    "Keep each step to 1–2 short sentences.\n\nAnswer:"
)

PROMPT_TEMPLATE_NORMAL = (
    "You are an expert summarizer of sports events. Use ONLY the following context:\n\n{context}\n\n"
    "Question: {query}\n\n"
    "Answer in two parts:\n"
    "1. A concise final result/recommendation in one paragraph (50–100 words).\n"
    "2. A numbered list of 3–6 concrete, externally-actionable steps. For each step include:\n"
    "   (a) short description, (b) difficulty (low/medium/high), (c) required inputs/resources.\n"
    "   If an input/resource is not specified, mark it 'MISSING: <name>'.\n"
    "Do NOT show internal reasoning. Only output the final answer.\n\nAnswer:"
)

PROMPT_TEMPLATE_EVAL = """You are an expert evaluator. Using ONLY the CONTEXT below, judge whether the claims made in the MODEL_ANSWER are supported by the CONTEXT.

Return ONLY a valid JSON object wrapped inside a markdown json code block (i.e. ```json ... ```). The JSON must contain the keys:
 - "precision": float (0.0-1.0)
 - "recall": float (0.0-1.0)
 - "f1": float
 - "supported_claims": int
 - "total_claims": int
 - "missing_details": list[str]
 - "rationale": str

IMPORTANT: Do NOT output any prose outside the ```json``` block. If you cannot compute a metric, use null for that value.

CONTEXT:
{context}

MODEL_ANSWER:
{answer}

OUTPUT:
"""  # will be formatted with .format(context=context, answer=answer)

# ---------------- Utilities ----------------

logger = logging.getLogger(__name__)


def safe_normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_words(text: str) -> List[str]:
    text = safe_normalize_text(text)
    return text.split()


def remove_repeated_sentences(text: str) -> str:
    text = text or ""
    try:
        if nltk:
            sentences = nltk.tokenize.sent_tokenize(text)
        else:
            raise Exception("nltk not available")
    except Exception:
        sentences = [s.strip() for s in re.split(r"[\n\.]+'", text) if s.strip()]
    seen = set()
    out = []
    for s in sentences:
        s_norm = s.strip()
        if s_norm and s_norm not in seen:
            out.append(s_norm)
            seen.add(s_norm)
    return ". ".join(out).strip()


def collapse_adjacent_duplicate_phrases(text: str) -> str:
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    pattern = re.compile(r"(.{10,200}?)\s+\1", flags=re.IGNORECASE | re.DOTALL)
    for _ in range(3):
        text = pattern.sub(r"\1", text)
    return text.strip()


def clean_model_output(text: str) -> str:
    text = text or ""
    text = remove_repeated_sentences(text)
    text = collapse_adjacent_duplicate_phrases(text)
    return text.strip()


def evaluate_response_detailed(pred: str, gold: str) -> Dict[str, Any]:
    pred_tokens = set(tokenize_words(pred))
    gold_tokens = set(tokenize_words(gold))
    tp = len(pred_tokens & gold_tokens)
    fp = len(pred_tokens - gold_tokens)
    fn = len(gold_tokens - pred_tokens)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall, "f1": f1}


# ---------------- Cached resources ----------------

@st.cache_resource
def get_device_index() -> int:
    """Return the device index expected by transformers pipeline.
    - GPU: 0
    - CPU: -1
    """
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


@st.cache_resource
def load_rag_pipeline(model_name: str = RAG_MODEL, max_length: int = MAX_GENERATION_LENGTH):
    """Load and return a Hugging Face text2text-generation pipeline.
    Returns the raw pipeline object, not wrapped by langchain, for predictable outputs.
    """
    if TRANSFORMERS_IMPORT_ERROR:
        raise RuntimeError(f"transformers import failed: {TRANSFORMERS_IMPORT_ERROR}")

    device = get_device_index()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        # Provide an informative error for common pitfalls (no internet, large model)
        raise RuntimeError(
            f"Failed to load model '{model_name}'. This may be because the model is large or you have no internet access. "
            f"Original error: {e}"
        )

    hf_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
        do_sample=False,
        num_beams=4,
        early_stopping=True,
    )
    return hf_pipe


@st.cache_resource
def get_embedder(model_name: str = EMBED_MODEL):
    if LANGCHAIN_IMPORT_ERROR:
        raise RuntimeError(f"langchain community imports failed: {LANGCHAIN_IMPORT_ERROR}")
    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embedder '{model_name}': {e}")


# ---------------- Document helpers ----------------

def load_documents_from_folder(folder: Path = DEFAULT_DOCS_DIR) -> List:
    if LANGCHAIN_IMPORT_ERROR:
        st.error("LangChain imports failed on this environment. Details: " + str(LANGCHAIN_IMPORT_ERROR))
        return []

    if TextLoader is None:
        st.error("TextLoader is not available in this environment. Make sure you have 'langchain' or 'langchain-community' installed.")
        return []

    docs: List = []
    folder = Path(folder)
    if not folder.exists():
        st.warning(f"Docs folder '{folder}' does not exist. Create it and add .txt files.")
        return docs
    txt_files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if not txt_files:
        st.warning(f"No .txt files found in '{folder}'.")
        return docs
    for p in txt_files:
        try:
            loader = TextLoader(str(p), encoding="utf-8")
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            st.warning(f"Skipping {p.name}: {e}")
    st.success(f"Loaded {len(docs)} documents from {len(txt_files)} files.")
    return docs


def split_documents(documents: List, chunk_size: int = SPLIT_CHUNK_SIZE, chunk_overlap: int = SPLIT_CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


@st.cache_resource
def create_chroma_from_documents(_docs, persist_dir: str = str(CHROMA_PERSIST_DIR)):
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    embedder = get_embedder()
    # Chroma accepts documents and the embedding function
    db = Chroma.from_documents(_docs, embedder, persist_directory=str(persist_dir))
    db.persist()
    return db


def load_chroma(persist_dir: Path = CHROMA_PERSIST_DIR, embedder=None):
    if LANGCHAIN_IMPORT_ERROR:
        raise RuntimeError(f"langchain community imports failed: {LANGCHAIN_IMPORT_ERROR}")
    persist_dir = Path(persist_dir)
    embedder = embedder or get_embedder()
    if not persist_dir.exists():
        return None
    return Chroma(persist_directory=str(persist_dir), embedding_function=embedder)


# ---------------- Evaluation with LLM ----------------


def _try_parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from model output.
    Steps:
    1) Prefer an explicit ```json { ... } ``` code block and parse it.
    2) Fallback to the first { ... } block and try lenient fixes (single->double quotes, remove trailing commas).
    3) As a last resort, extract numeric metrics using regex (e.g., 'Precision: 0.8, Recall: 0.6, F1: 0.7').
    Returns parsed dict on success, otherwise None.
    """
    if not text:
        return None

    # 1) look for explicit ```json { ... } ``` block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        js = m.group(1)
        try:
            return json.loads(js)
        except Exception:
            pass

    # 2) fallback to any {...} block
    m2 = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m2:
        js = m2.group(1)
        try:
            return json.loads(js)
        except Exception:
            # minor fixes: single quotes -> double, remove trailing commas
            fixed = js.replace("'", '"')
            fixed = re.sub(r",\s*\}", "}", fixed)
            fixed = re.sub(r",\s*\]", "]", fixed)
            try:
                return json.loads(fixed)
            except Exception:
                pass

    # 3) regex numeric extraction: Precision: 0.8, Recall=0.6, F1: 0.7
    numeric = {}
    m_prec = re.search(r"precision\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    m_rec = re.search(r"recall\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    m_f1 = re.search(r"f1\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    m_sup = re.search(r"supported[_\s]claims\s*[:=]\s*([0-9]+)", text, flags=re.IGNORECASE)
    m_tot = re.search(r"total[_\s]claims\s*[:=]\s*([0-9]+)", text, flags=re.IGNORECASE)

    if m_prec:
        try:
            numeric['precision'] = float(m_prec.group(1))
        except Exception:
            pass
    if m_rec:
        try:
            numeric['recall'] = float(m_rec.group(1))
        except Exception:
            pass
    if m_f1:
        try:
            numeric['f1'] = float(m_f1.group(1))
        except Exception:
            pass
    if m_sup:
        try:
            numeric['supported_claims'] = int(m_sup.group(1))
        except Exception:
            pass
    if m_tot:
        try:
            numeric['total_claims'] = int(m_tot.group(1))
        except Exception:
            pass

    if numeric:
        # fill missing keys with None / empty defaults
        result = {
            'precision': numeric.get('precision'),
            'recall': numeric.get('recall'),
            'f1': numeric.get('f1'),
            'supported_claims': numeric.get('supported_claims'),
            'total_claims': numeric.get('total_claims'),
            'missing_details': [],
            'rationale': None
        }
        return result

    return None


def evaluate_with_llm(llm_pipe, answer: str, context: str) -> Dict[str, Any]:
    """Ask the LLM to act as a judge and return precision/recall/f1 + rationale.
    Returns a dict with keys: precision, recall, f1, supported_claims, total_claims, missing_details, rationale
    If parsing fails, returns an entry 'parsing_failed' and the raw output.
    """
    prompt = PROMPT_TEMPLATE_EVAL.format(context=context, answer=answer)
    try:
        outputs = llm_pipe(prompt, num_return_sequences=1, max_length=512)
        raw = outputs[0].get("generated_text", "") if isinstance(outputs, list) and outputs else str(outputs)
        parsed = _try_parse_json_from_text(raw)
        if parsed:
            # normalize numeric values
            for k in ["precision", "recall", "f1"]:
                if k in parsed:
                    try:
                        parsed[k] = float(parsed[k]) if parsed[k] is not None else None
                    except Exception:
                        parsed[k] = None
            return {**parsed, "raw": raw}
        else:
            return {"parsing_failed": True, "raw": raw}
    except Exception as e:
        return {"error": str(e)}


# ---------------- Streamlit UI and flow ----------------

st.set_page_config(page_title="RAG Sports Event QA App (LLM Judge)", layout="centered")
st.title("🔍 RAG: Sports Event Report App — LLM Judge")
st.caption("Retrieval-Augmented Generation for sports event QA. Load docs -> build index -> ask questions.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    k_results = st.number_input("Retriever k (top docs)", min_value=1, max_value=10, value=DEFAULT_K)
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=5000, value=SPLIT_CHUNK_SIZE)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=SPLIT_CHUNK_OVERLAP)
    model_name = st.text_input("RAG model name (HuggingFace)", value=RAG_MODEL)
    embed_model_name = st.text_input("Embedding model name", value=EMBED_MODEL)
    regenerate_index = st.button("🔁 Rebuild index (use after changing docs or chunk settings)")
    clear_db = st.button("🗑️ Clear vector DB")

# Session state
if "db_loaded" not in st.session_state:
    st.session_state["db_loaded"] = False
if "last_result" not in st.session_state:
    st.session_state["last_result"] = ""
if "metrics" not in st.session_state:
    st.session_state["metrics"] = None
if "llm_judge" not in st.session_state:
    st.session_state["llm_judge"] = None

# Top-level buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("📄 Load Documents"):
        try:
            docs = load_documents_from_folder(DEFAULT_DOCS_DIR)
            st.session_state["loaded_docs_count"] = len(docs)
        except Exception as e:
            st.error(f"Error while loading documents: {e}")
with col2:
    if regenerate_index:
        try:
            docs = load_documents_from_folder(DEFAULT_DOCS_DIR)
            if docs:
                split_docs = split_documents(docs, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
                create_chroma_from_documents(split_docs, persist_dir=str(CHROMA_PERSIST_DIR))
                st.session_state["db_loaded"] = True
                st.success("✅ Documents processed and saved to vector DB.")
            else:
                st.warning("No documents to index. Add .txt files into the docs folder.")
        except Exception as e:
            st.error(f"Failed to build index: {e}")

if clear_db:
    try:
        if CHROMA_PERSIST_DIR.exists():
            for p in CHROMA_PERSIST_DIR.iterdir():
                if p.is_file():
                    p.unlink()
            CHROMA_PERSIST_DIR.rmdir()
        st.session_state["db_loaded"] = False
        st.success("Vector DB cleared.")
    except Exception as e:
        st.error(f"Could not clear DB: {e}")

# Quick status
st.write("---")
if CHROMA_PERSIST_DIR.exists():
    st.info(f"Vector DB directory: {CHROMA_PERSIST_DIR} (exists)")
else:
    st.info(f"Vector DB directory: {CHROMA_PERSIST_DIR} (not found) - build index to create it")

# Query input area
query = st.text_input("🔎 Enter your question about the sports event:", key="query_input")
prompting_style = st.radio("🧠 Choose Prompting Style", ["Normal", "Chain of Thought"], horizontal=True)

if st.button("Ask"):
    if LANGCHAIN_IMPORT_ERROR:
        st.error("Required langchain_community modules are not installed or failed to import.\n"
                 f"Original import error: {LANGCHAIN_IMPORT_ERROR}")
    elif TRANSFORMERS_IMPORT_ERROR:
        st.error("transformers library not available or failed to import.\n"
                 f"Original import error: {TRANSFORMERS_IMPORT_ERROR}")
    elif not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Searching for answers..."):
            try:
                # Load resources
                embedder = get_embedder(model_name=embed_model_name)
                db = load_chroma(CHROMA_PERSIST_DIR, embedder=embedder)
                if db is None:
                    st.error("Vector DB not available. Build index first (use Rebuild index button).")
                else:
                    # Load LLM pipeline lazily (may take some time)
                    llm_pipe = load_rag_pipeline(model_name=model_name, max_length=MAX_GENERATION_LENGTH)

                    retriever = db.as_retriever(search_kwargs={"k": int(k_results)})
                    # Use method name compatible with langchain retriever
                    get_docs_fn = getattr(retriever, "get_relevant_documents", None) or getattr(retriever, "get_documents", None)
                    if get_docs_fn is None:
                        raise RuntimeError("Retriever does not expose a compatible get_relevant_documents method.")

                    docs = get_docs_fn(query.strip())
                    if not docs:
                        st.warning("No relevant documents found for the query.")
                        st.session_state["last_result"] = ""
                    else:
                        context = "\n\n".join([d.page_content for d in docs])
                        if prompting_style == "Chain of Thought":
                            final_prompt = PROMPT_TEMPLATE_COT.format(query=query.strip(), context=context)
                        else:
                            final_prompt = PROMPT_TEMPLATE_NORMAL.format(query=query.strip(), context=context)

                        # Call pipeline: returns list[dict({'generated_text': ...})]
                        outputs = llm_pipe(final_prompt, num_return_sequences=1)
                        if isinstance(outputs, list) and outputs:
                            raw_answer = outputs[0].get("generated_text", "")
                        elif isinstance(outputs, dict):
                            raw_answer = outputs.get("generated_text", "")
                        else:
                            raw_answer = str(outputs)

                        cleaned_result = clean_model_output(raw_answer)

                        st.session_state["last_result"] = cleaned_result

                        # Show retrieved snippets
                        st.markdown("**Retrieved source snippets (top k):**")
                        for i, doc in enumerate(docs[: int(k_results)], 1):
                            snippet = doc.page_content.strip().replace("\n", " ")
                            st.write(f"{i}. {snippet[:700]}{'...' if len(snippet) > 700 else ''}")

                        # --- Automatic evaluation using LLM judge ---
                        eval_res = evaluate_with_llm(llm_pipe, cleaned_result, context)
                        if eval_res.get('error'):
                            st.session_state['llm_judge'] = None
                        elif eval_res.get('parsing_failed'):
                            # Do not store or display raw non-JSON output; skip LLM judge display
                            st.session_state['llm_judge'] = None
                        else:
                            # parsed JSON
                            st.session_state['llm_judge'] = eval_res

                        # Always compute token-overlap fallback metrics (context used as 'gold')
                        token_metrics = evaluate_response_detailed(cleaned_result, context)
                        st.session_state['metrics'] = token_metrics

            except Exception as e:
                st.error(f"Error during retrieval/generation: {e}")

# Display answer + evaluation
if st.session_state.get("last_result"):
    st.write("---")
    st.write("📢 **Answer:**")
    st.info(st.session_state["last_result"])

    # Removed: manual ground-truth expected answer input (user requested removal)

    # Show LLM judge results if available
    if st.session_state.get('llm_judge'):
        st.write('### 🤖 LLM Judge Results')
        lj = st.session_state['llm_judge']
        # only show parsed, trusted JSON metrics (no raw output shown anywhere)
        p = lj.get('precision')
        r = lj.get('recall')
        f = lj.get('f1')
        supported = lj.get('supported_claims')
        total = lj.get('total_claims')
        missing = lj.get('missing_details')
        rationale = lj.get('rationale')
        st.write(f"- **Precision (LLM judge):** {p if p is not None else 'N/A'}")
        st.write(f"- **Recall (LLM judge):** {r if r is not None else 'N/A'}")
        st.write(f"- **F1 (LLM judge):** {f if f is not None else 'N/A'}")
        st.write(f"- **Supported claims:** {supported if supported is not None else 'N/A'} / {total if total is not None else 'N/A'}")
        if missing:
            st.write(f"- **Missing details:** {missing}")
        if rationale:
            st.write(f"- **Rationale:** {rationale}")

    # Show token-overlap computed metrics
    if st.session_state.get("metrics"):
        m = st.session_state["metrics"]
        st.write("### 📝 Token-overlap Metrics (fallback)")
        st.write(f"- **Precision:** {m['precision']:.2f}")
        st.write(f"- **Recall:** {m['recall']:.2f}")
        st.write(f"- **F1 Score:** {m['f1']:.2f}")
        st.info(
            f"- **True Positives:** {m['true_positive']}\n"
            f"- **False Positives:** {m['false_positive']}\n"
            f"- **False Negatives:** {m['false_negative']}"
        )

