import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import base64
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai


# ---------------- CONFIG ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API")
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API in environment. Put your Google API key in a .env file as GOOGLE_API or set env var.")
genai.configure(api_key=GOOGLE_API_KEY)

# Path to folder containing your internal PDFs (keeps internal knowledge)
DATA_FILE = Path(r"D:\RAG-Project\documents")  # change if needed
DB_DIR = Path("vectorstore")  # folder where FAISS index will be saved

# Vector/embedding config
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_FOLDER = Path(".cache/hf")

# RAG system prompt (you had this; adjust as needed)
SYSTEM_PROMPT = """
You are a certified expert mixologist with advanced knowledge of spirits, cocktail theory, preparation methods, flavor pairing, and bar techniques.
Use only the information contained in the retrieved context to answer the user’s question.
Carefully analyze the context, extract the most relevant details, and construct a clear, accurate answer.
If the context does not provide enough information to answer the question, explicitly state that you do not know.
Present your reasoning and instructions in a clear, step-by-step format that is easy for the user to follow.
"""

# ---------------- HELPERS ----------------
def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

def make_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=str(CACHE_FOLDER),
    )

def build_store_from_folder(folder: Path, save_to=DB_DIR):
    """Load PDFs from local folder, split, embed and create FAISS store."""
    if not folder.exists():
        raise FileNotFoundError(f"{folder} missing")

    all_docs = []
    for file in folder.iterdir():
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
            docs = loader.load()
            all_docs.extend(docs)

    splitter = make_splitter()
    chunks = splitter.split_documents(all_docs)

    embeddings = make_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    save_to.mkdir(exist_ok=True)
    vectorstore.save_local(str(save_to))
    return vectorstore

def load_store(db_dir=DB_DIR):
    """Load FAISS store from disk."""
    if not db_dir.exists():
        raise FileNotFoundError(f"{db_dir} not found")
    embeddings = make_embeddings()
    return FAISS.load_local(str(db_dir), embeddings, allow_dangerous_deserialization=True)

def add_uploaded_pdf_to_store(uploaded_file, vectorstore):
    """Save the uploaded file temporarily, load pages, split, and add to vectorstore."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    loader = PyPDFLoader(str(tmp_path))
    docs = loader.load()
    splitter = make_splitter()
    chunks = splitter.split_documents(docs)

    # Add chunks to existing vectorstore
    vectorstore.add_documents(chunks)
    # optional: save new combined vectorstore
    DB_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(DB_DIR))
    # cleanup temp file
    try:
        tmp_path.unlink()
    except Exception:
        pass
    return len(chunks)

def query_vectorstore_and_answer(vectorstore, user_query, k=5):
    docs = vectorstore.similarity_search(user_query, k=k)
    context_text = "\n\n---\n\n".join(d.page_content for d in docs) if docs else ""

    prompt = [
        SYSTEM_PROMPT,
        "\n\nRetrieved context:\n",
        context_text if context_text else "[No matching context found]",
        "\n\nUser question:\n",
        user_query,
    ]

    model = genai.GenerativeModel("gemini-2.5-flash")
    # The genai python API may accept different shapes; we try to pass a list-of-strings (like your code)
    # If your installed genai version requires a dict e.g. {"prompt": prompt_str}, convert accordingly.
    response = model.generate_content(prompt)
    # response object shape may vary -- try to use `.text` or `.response_text` or inspect in logs.
    # We'll defensively pull whichever attribute exists.
    text = None
    if hasattr(response, "text"):
        text = response.text
    elif hasattr(response, "response_text"):
        text = response.response_text
    elif isinstance(response, dict):
        # attempt common keys
        text = response.get("output", "") or response.get("candidates", [{}])[0].get("content", "")
    else:
        text = str(response)

    return text, docs

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="RAG PDF Question Answering", layout="wide")
# ---------------- BACKGROUND IMAGE (base64) ----------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Optional dark overlay for readability */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.35);
        z-index: 0;
    }}

    /* Ensure real content sits above overlay */
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """

    st.markdown(bg_css, unsafe_allow_html=True)

# Call it
set_bg("op.jpg")


st.title(" Mixology RAG Assistant — Ask, Retrieve, Sip Knowledge")


st.sidebar.header("Settings")
k = st.sidebar.slider("Number of retrieved chunks (k)", min_value=1, max_value=10, value=5)
rebuild_index = st.sidebar.button("Rebuild index from internal folder")

# Session state for vectorstore
if "vectorstore" not in st.session_state:
    # Try to load existing index, otherwise build from local folder if present
    try:
        st.session_state.vectorstore = load_store()
        st.sidebar.success("Loaded existing vectorstore from disk.")
    except Exception as e:
        try:
            st.sidebar.info("No saved vectorstore found — building from internal folder.")
            st.session_state.vectorstore = build_store_from_folder(DATA_FILE)
            st.sidebar.success("Built vectorstore from internal PDFs.")
        except Exception as build_e:
            st.session_state.vectorstore = None
            st.sidebar.error(f"Could not build/load vectorstore: {build_e}")

# Rebuild index manually
if rebuild_index:
    try:
        st.session_state.vectorstore = build_store_from_folder(DATA_FILE)
        st.sidebar.success("Rebuilt vectorstore from internal folder.")
    except Exception as e:
        st.sidebar.error(f"Failed to rebuild: {e}")

# Upload PDF
st.subheader("Upload a PDF to include in the search (optional)")
uploaded = st.file_uploader("Upload a PDF (it will be merged into the index for this session)", type=["pdf"])

if uploaded is not None:
    if st.session_state.vectorstore is None:
        st.error("Vectorstore not ready. Try rebuilding the index from internal folder first.")
    else:
        with st.spinner("Adding uploaded PDF to vectorstore..."):
            try:
                added = add_uploaded_pdf_to_store(uploaded, st.session_state.vectorstore)
                st.success(f"Uploaded PDF added to vectorstore — {added} chunks indexed.")
            except Exception as e:
                st.error(f"Failed to add uploaded PDF: {e}")

# Query UI
st.subheader("Ask a question (answers will be based on retrieved context + internal docs)")
user_query = st.text_area("Enter your question here", height=120)

if st.button("Get Answer"):
    if not user_query or st.session_state.vectorstore is None:
        st.warning("Please provide a question and ensure vectorstore is ready.")
    else:
        with st.spinner("Retrieving context and calling the model..."):
            try:
                answer_text, retrieved_docs = query_vectorstore_and_answer(st.session_state.vectorstore, user_query, k=k)
                st.markdown("### Answer")
                st.write(answer_text)

                with st.expander("Retrieved contexts (for transparency)"):
                    for i, d in enumerate(retrieved_docs, start=1):
                        st.markdown(f"**Chunk {i} — source:** {getattr(d, 'metadata', {}).get('source', 'unknown')}")
                        st.write(d.page_content[:1000])  # show first 1000 chars
            except Exception as e:
                st.error(f"Error while querying/generating: {e}")

st.markdown("---")
st.caption("Notes: Uploaded PDFs are temporarily saved and then deleted. The combined vectorstore is saved to `vectorstore/` on disk.")
