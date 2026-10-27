from pathlib import Path


def test_requirements_file_has_core_dependencies():
    candidates = [Path("requirements.txt"), Path("requiremets.txt")]
    file_path = next((path for path in candidates if path.exists()), candidates[0])
    text = file_path.read_text(encoding="utf-8")

    assert "streamlit" in text.lower()
    assert "langchain" in text.lower()
    assert "pypdf" in text.lower()
    assert "python-dotenv" in text.lower()
