import os
import argparse
import whisper
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize

# --- IMPORTS ---
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

# --- CONFIGURATION ---
BASE_URL = "http://localhost:1234/v1"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "agent_db"

# LLM & Embeddings
local_llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key="lm-studio",
    model="hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    base_url=BASE_URL,
    api_key="lm-studio",
    model="second-state/All-MiniLM-L6-v2-Embedding-GGUF",
    check_embedding_ctx_length=False
)

client = QdrantClient(url=QDRANT_URL)
search_tool = DuckDuckGoSearchRun()


# --- PART 1: DATA INGESTION ---

def extract_pdf_text(path):
    if not os.path.exists(path): return ""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"
    return text


def transcribe_audio(path):
    if not os.path.exists(path): return ""
    model = whisper.load_model("base")
    return model.transcribe(path)["text"]


def chunk_text(text, chunk_size=300, overlap=50):
    if not text: return []
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = " ".join(current_chunk.split()[-overlap:]) + " " + sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks


def ingest_data(pdf_path=None, audio_path=None):
    all_text = ""
    if pdf_path: all_text += extract_pdf_text(pdf_path)
    if audio_path: all_text += transcribe_audio(audio_path)
    if not all_text.strip(): return

    chunks = chunk_text(all_text)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    vector_store = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings, content_payload_key="page_content"
    )
    vector_store.add_texts(chunks)
    print("Ingestion Complete.")


# --- PART 2: AGENT STATE ---

def create_initial_state(question):
    return {
        "question": question,
        "documents": [],
        "answer": "",
        "source": "",
        "relevance": "no",
        "eval_score": 0
    }


# --- PART 3: FUNCTIONAL TOOLS ---

def save_to_file_tool(question, answer, source):
    print("--- 💾 SAVING TO FILE ---")
    """
    Saves the QA pair to a file.
    Now strictly a helper function, not called inside other logic.
    """
    try:
        with open("agent_output.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 20}\nSOURCE: {source}\nQ: {question}\nA: {answer}\n")
        print("--- 💾 TOOL CALLED: SAVED TO FILE ---")
    except Exception as e:
        print(f"Error saving file: {e}")


def retrieve(state):
    print(f"--- 🔍 RETRIEVING: {state['question']} ---")
    if not client.collection_exists(COLLECTION_NAME):
        state["documents"] = []
        return state

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings,
                                     content_payload_key="page_content")
    docs = vector_store.as_retriever(search_kwargs={"k": 3}).invoke(state["question"])
    state["documents"] = [d.page_content for d in docs]
    return state


def grade_documents(state):
    print("--- 🧠 GRADING DOCUMENTS ---")
    if not state["documents"]:
        state["relevance"] = "no"
        return state

    prompt = ChatPromptTemplate.from_template(
        "Is this document relevant to: {question}? Doc: {context}. Return ONLY 'yes' or 'no'."
    )
    chain = prompt | local_llm | StrOutputParser()
    score = chain.invoke({"question": state["question"], "context": state["documents"][0]})

    state["relevance"] = "yes" if "yes" in score.lower() else "no"
    print(f"--- GRADE: {state['relevance'].upper()} ---")
    return state


def generate_rag(state):
    print("--- 💡 GENERATING RAG ANSWER ---")
    prompt = ChatPromptTemplate.from_template("Answer based on context: {context}. Q: {question}")
    chain = prompt | local_llm | StrOutputParser()
    answer = chain.invoke({"question": state["question"], "context": "\n".join(state["documents"])})

    # REMOVED: save_to_file_tool(...) call

    state["answer"] = answer
    state["source"] = "rag"
    return state


def perform_web_search(state):
    print("--- 🌍 WEB SEARCH (Step 1/2) ---")
    try:
        results = search_tool.invoke(state["question"])
    except:
        results = "No results found."

    prompt = ChatPromptTemplate.from_template("Answer based on web results: {context}. Q: {question}")
    answer = (prompt | local_llm | StrOutputParser()).invoke({"question": state["question"], "context": results})

    # REMOVED: save_to_file_tool(...) call

    state["answer"] = answer
    state["source"] = "web"
    return state


def generate_social_post(state):
    print("--- 📢 GENERATING SOCIAL POST (Step 2/2) ---")
    # Kept your simple string concatenation logic
    state["answer"] += "\n\n--- [GENERATED SOCIAL POST] ---\n[Fun with #Ciklum and #CiklumAiAcademy!]\n" + state["answer"]
    return state


def evaluate_answer(state):
    print("--- 📊 EVALUATING RESULT ---")
    prompt = ChatPromptTemplate.from_template(
        "Rate relevance 1-5. Q: {question} A: {answer}. Return ONLY integer."
    )
    chain = prompt | local_llm | StrOutputParser()
    try:
        score = int(chain.invoke({"question": state["question"], "answer": state["answer"]}).strip())
    except:
        score = 3
    print(f"--- SCORE: {score}/5 ---")
    state["eval_score"] = score
    return state


# --- PART 4: THE ORCHESTRATOR ---

def run_agent_workflow(question):
    # 1. Initialize
    state = create_initial_state(question)

    # 2. Retrieve & Grade
    state = retrieve(state)
    state = grade_documents(state)

    # 3. Decision Logic
    if state["relevance"] == "yes":
        # --- RAG PATH ---
        state = generate_rag(state)

        # ✅ EXPLICIT SAVE (Orchestration level)
        save_to_file_tool(state["question"], state["answer"], "RAG")

        state = evaluate_answer(state)

        # 4. Retry Loop
        if state["eval_score"] < 3:
            print("--- 🔄 LOW RAG SCORE. RETRYING WITH WEB... ---")

            # --- WEB PATH (Retry) ---
            state = perform_web_search(state)

            # ✅ EXPLICIT SAVE (Orchestration level)
            save_to_file_tool(state["question"], state["answer"], "WEB (RETRY)")

            state = generate_social_post(state)
            state = evaluate_answer(state)

    else:
        # --- WEB PATH (Direct) ---
        print("--- DOCUMENTS IRRELEVANT. SWITCHING TO WEB. ---")
        state = perform_web_search(state)

        # ✅ EXPLICIT SAVE (Orchestration level)
        save_to_file_tool(state["question"], state["answer"], "WEB (DIRECT)")

        state = generate_social_post(state)
        state = evaluate_answer(state)

    return state


# --- EXECUTION ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str)
    parser.add_argument("--audio", type=str)
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    if args.pdf: ingest_data(pdf_path=args.pdf)

    if args.audio: ingest_data(audio_path=args.audio)

    if args.query:
        print(f"\n🚀 STARTING AGENT FOR: '{args.query}'\n")

        # Run the Python-only orchestrator
        final_state = run_agent_workflow(args.query)

        print(f"\nFINAL ANSWER:\n{final_state['answer']}")