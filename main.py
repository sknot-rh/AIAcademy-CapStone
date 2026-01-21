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
COLLECTION_NAME = "rag_db"
VECTOR_SIZE = 384
MAX_RETRIES = 2

# LLM & Embeddings
local_llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key="lm-studio",
    model="hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF",
    temperature=0.7
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
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
    )

    vector_store = QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings, content_payload_key="page_content"
    )
    vector_store.add_texts(chunks)
    print("✅ Ingestion Complete.")


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


# --- PART 3: CORE FUNCTIONS ---

def save_to_file_tool(question, answer, source):
    try:
        with open("agent_output.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 20}\nSOURCE: {source}\nQUESTION: {question}\nANSWER: {answer}\n")
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
    try:
        docs = vector_store.as_retriever(search_kwargs={"k": 3}).invoke(state["question"])
        state["documents"] = [d.page_content for d in docs]
    except:
        print("Error during retrieval. As a fallback, no documents found.")
        state["documents"] = []
    return state


def grade_documents(state):
    print("--- 🧠 GRADING DOCUMENTS ---")
    if not state["documents"]:
        state["relevance"] = "no"
        return state

    # ✅ BALANCED PROMPT: Checks for concepts/topic match, allowing for imperfect docs.
    prompt = ChatPromptTemplate.from_template(
        """You are a relevance grader.

        Question: {question}
        Document: {context}

        Does this document contain keywords or concepts related to the question? 
        It does not need to be a perfect answer. If it's the same topic, say 'yes'.

        Return ONLY 'yes' or 'no'."""
    )
    chain = prompt | local_llm | StrOutputParser()
    score = chain.invoke({"question": state["question"], "context": state["documents"][0]})

    is_relevant = "yes" in score.lower()[:20]
    state["relevance"] = "yes" if is_relevant else "no"
    print(f"--- GRADE: {state['relevance'].upper()} ---")
    return state


def generate_rag(state):
    print("--- 💡 GENERATING RAG ANSWER ---")
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Use the context to answer the question.
        Context: {context}
        Question: {question}

        If the context doesn't help, strictly say "I don't know".
        """
    )
    chain = prompt | local_llm | StrOutputParser()
    answer = chain.invoke({"question": state["question"], "context": "\n".join(state["documents"])})
    state["answer"] = answer
    state["source"] = "rag"
    return state


def _perform_single_web_search(state):
    print("--- 🌍 WEB SEARCH (Fetching...) ---")
    try:
        results = search_tool.invoke(state["question"])
    except:
        print("Error during web search. Returning 'No results found'.")
        results = "No results found."

    prompt = ChatPromptTemplate.from_template("Answer based on web results: {context}. QUESTION: {question}")
    answer = (prompt | local_llm | StrOutputParser()).invoke({"question": state["question"], "context": results})
    state["answer"] = answer
    state["source"] = "web"
    return state


def generate_social_post(state):
    print("--- 📢 GENERATING SOCIAL POST ---")
    state["answer"] += "\n\n--- [GENERATED SOCIAL POST] ---\n[Fun with #Ciklum and #CiklumAiAcademy!]\n" + state[
        "answer"]
    return state


def evaluate_answer(state):
    print("--- 📊 EVALUATING RESULT ---")
    # Checks if the answer addresses the question directly
    prompt = ChatPromptTemplate.from_template(
        """Review the answer.
        Question: {question}
        Answer: {answer}

        Does the answer directly address the specific question asked?
        If it talks about a different topic or says "I don't know", score it 1.

        Rate relevance 1-5. Return ONLY integer."""
    )
    chain = prompt | local_llm | StrOutputParser()
    str_val = ""
    try:
        str_val = chain.invoke({"question": state["question"], "answer": state["answer"]}).strip()
        score = int(str_val)
    except:
        print("Error parsing score (" + str_val + "). Defaulting to 3.")
        score = 3
    print(f"--- SCORE: {score}/5 ---")
    state["eval_score"] = score
    return state


# --- PART 4: ORCHESTRATOR ---

def run_web_search_with_retry(state):
    attempt = 1
    while attempt <= MAX_RETRIES:
        print(f"\n--- 🔄 WEB ATTEMPT {attempt}/{MAX_RETRIES} ---")
        state = _perform_single_web_search(state)
        state = evaluate_answer(state)

        if state["eval_score"] >= 3:
            print(f"--- ✅ SUCCESS on attempt {attempt} ---")
            return state

        print(f"--- ⚠️ ATTEMPT {attempt} FAILED (Score {state['eval_score']}). Retrying... ---")
        attempt += 1

    print("--- ❌ ALL RETRIES EXHAUSTED ---")
    return state


def run_agent_workflow(question):
    state = create_initial_state(question)
    state = retrieve(state)
    state = grade_documents(state)

    if state["relevance"] == "yes":
        state = generate_rag(state)
        state = evaluate_answer(state)

        # Retry if score is low OR if the model explicitly gave up
        if state["eval_score"] < 3 or "don't know" in state["answer"].lower():
            print(f"--- 📉 RAG FAILED (Score {state['eval_score']}). SWITCHING TO WEB... ---")
            state["source"] = "web (fallback)"
            state = run_web_search_with_retry(state)
    else:
        print("--- 📄 DOCUMENTS IRRELEVANT. STARTING WEB LOOP... ---")
        state["source"] = "web (direct)"
        state = run_web_search_with_retry(state)

    if state["eval_score"] >= 3:
        save_to_file_tool(state["question"], state["answer"], state["source"])
        state = generate_social_post(state)
    else:
        print(f"\n--- 🛑 FINAL SCORE {state['eval_score']} TOO LOW. SKIPPING SAVE & POST. ---")

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
        final_state = run_agent_workflow(args.query)
        print(f"\nFINAL ANSWER:\n{final_state['answer']}")