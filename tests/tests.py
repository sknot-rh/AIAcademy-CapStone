import pytest
from unittest.mock import MagicMock, patch, mock_open
import main  # Assumes your script is named main.py


# --- 1. Tests for chunk_text (Logic Check) ---

def test_chunk_text_empty():
    """Test that empty or None input returns an empty list."""
    assert main.chunk_text("") == []
    assert main.chunk_text(None) == []
    print("Test for chunk_text_empty passed.")


def test_chunk_text_short():
    """Test that text shorter than chunk_size is returned as a single chunk."""
    text = "This is a short sentence."
    # chunk_size is large enough to hold the whole text
    chunks = main.chunk_text(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == text
    print("Test for chunk_text_short passed.")


def test_chunk_text_splitting():
    """Test that long text is actually split into multiple chunks."""
    # Create a long string with identifiable sentences
    text = "Sentence one. " * 50  # 100 words approx
    chunks = main.chunk_text(text, chunk_size=10, overlap=0)
    assert len(chunks) > 1
    print("Test for chunk_text_splitting passed.")


# --- 2. Tests for extract_pdf_text (File I/O) ---

def test_extract_pdf_text_not_found():
    """Test that a non-existent file returns an empty string."""
    assert main.extract_pdf_text("non_existent_ghost_file.pdf") == ""
    print("Test for extract_pdf_text_not_found passed.")


@patch("main.pdfplumber.open")
@patch("os.path.exists")
def test_extract_pdf_text_success(mock_exists, mock_pdf_open):
    """Test successful text extraction from a mocked PDF."""
    mock_exists.return_value = True

    # Mock the page object and its extract_text method
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Extracted PDF content."

    # Mock the PDF object context manager
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf

    result = main.extract_pdf_text("fake.pdf")
    assert "Extracted PDF content." in result
    print("Test for extract_pdf_text_success passed.")


# --- 3. Tests for grade_documents (Agent Logic) ---

def test_grade_documents_empty():
    """Test that having no documents returns 'no' immediately."""
    state = main.create_initial_state("query")
    state["documents"] = []

    result = main.grade_documents(state)
    assert result["relevance"] == "no"
    print("Test for empty documents passed.")


@patch("langchain_core.runnables.RunnableSequence.invoke")
def test_grade_documents_yes(mock_invoke):
    """Test that the grader correctly sets relevance to 'yes'."""
    # Mock LLM return value
    mock_invoke.return_value = "yes"

    state = main.create_initial_state("query")
    state["documents"] = ["Some relevant content"]

    result = main.grade_documents(state)
    assert result["relevance"] == "yes"
    print("Positive test for grading documents passed.")


@patch("langchain_core.runnables.RunnableSequence.invoke")
def test_grade_documents_no(mock_invoke):
    """Test that the grader correctly sets relevance to 'no'."""
    mock_invoke.return_value = "no"

    state = main.create_initial_state("query")
    state["documents"] = ["Irrelevant gibberish"]

    result = main.grade_documents(state)
    assert result["relevance"] == "no"
    print("Negative test for grading documents passed.")


# --- 4. Tests for evaluate_answer (Agent Logic) ---

@patch("langchain_core.runnables.RunnableSequence.invoke")
def test_evaluate_answer_high_score(mock_invoke):
    """Test parsing a valid high score."""
    mock_invoke.return_value = "5"

    state = main.create_initial_state("query")
    state["answer"] = "Perfect answer"

    result = main.evaluate_answer(state)
    assert result["eval_score"] == 5
    print("Test for high score passed.")


@patch("langchain_core.runnables.RunnableSequence.invoke")
def test_evaluate_answer_low_score(mock_invoke):
    """Test parsing a valid low score."""
    mock_invoke.return_value = "1"

    state = main.create_initial_state("query")
    state["answer"] = "I don't know"

    result = main.evaluate_answer(state)
    assert result["eval_score"] == 1
    print("Test for low score passed.")


@patch("langchain_core.runnables.RunnableSequence.invoke")
def test_evaluate_answer_parsing_failure(mock_invoke):
    """Test that it falls back to 3 if the LLM returns text instead of a number."""
    mock_invoke.return_value = "I think the score is 5"  # Not an int

    state = main.create_initial_state("query")
    state["answer"] = "Answer"

    result = main.evaluate_answer(state)
    # The try/except block in your code sets score = 3 on error
    assert result["eval_score"] == 3
    print("Test for parsing failure passed.")


# --- 5. Tests for State Management ---

def test_create_initial_state():
    """Test that state is initialized with correct keys."""
    state = main.create_initial_state("Is AI academy the best lesson?")
    assert state["question"] == "Is AI academy the best lesson?"
    assert state["documents"] == []
    assert state["relevance"] == "no"
    assert state["eval_score"] == 0