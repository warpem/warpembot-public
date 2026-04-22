import json
import os
import tempfile
import hashlib
import numpy as np
import pytest

from rag_common import message_id_to_hash, save_message, load_message


def test_message_id_to_hash():
    msg_id = "<CAL7Y77PkZWD9oYSjKRJjJe3tGv7D+7rNw5q0+WYQD_dA3x1Qzg@mail.gmail.com>"
    h = message_id_to_hash(msg_id)
    assert h == hashlib.sha256(msg_id.encode()).hexdigest()
    assert len(h) == 64


def test_save_and_load_message():
    with tempfile.TemporaryDirectory() as tmpdir:
        msg = {
            "message_id": "<test123@example.com>",
            "thread_id": "thread-1",
            "subject": "Test subject",
            "sender": "user@example.com",
            "date": "2025-01-15T10:30:00Z",
            "body": "Hello, this is a test message.",
        }
        path = save_message(msg, tmpdir)
        expected_hash = message_id_to_hash(msg["message_id"])
        assert os.path.basename(path) == f"{expected_hash}.json"

        loaded = load_message(path)
        assert loaded == msg


def test_save_message_dedup():
    """Saving the same message twice should not error and should return same path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        msg = {
            "message_id": "<test123@example.com>",
            "thread_id": "",
            "subject": "Test",
            "sender": "a@b.com",
            "date": "2025-01-01T00:00:00Z",
            "body": "Body text.",
        }
        path1 = save_message(msg, tmpdir)
        path2 = save_message(msg, tmpdir)
        assert path1 == path2


from unittest.mock import patch, MagicMock
from rag_common import get_embedding


def test_get_embedding_calls_ollama():
    """Mock the Ollama /api/embed endpoint. It returns {"embeddings": [[...]]}."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

    with patch("rag_common.requests.post", return_value=mock_response) as mock_post:
        vec = get_embedding("hello world")
        mock_post.assert_called_once()
        call_json = mock_post.call_args[1]["json"]
        assert call_json["model"] == "qwen3-embedding:8b"
        assert call_json["input"] == "hello world"
        assert len(vec) == 3
        # Check L2 normalization
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6


def test_get_embedding_with_instruction():
    """When an instruction is provided, the input should be prefixed with Instruct/Query format."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

    with patch("rag_common.requests.post", return_value=mock_response) as mock_post:
        vec = get_embedding("hello world", instruction="Retrieve relevant documents")
        call_json = mock_post.call_args[1]["json"]
        assert call_json["input"] == "Instruct: Retrieve relevant documents\nQuery: hello world"
        assert vec is not None


def test_get_embedding_without_instruction():
    """Without instruction, the input should be the raw text (for document embedding)."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

    with patch("rag_common.requests.post", return_value=mock_response) as mock_post:
        vec = get_embedding("hello world")
        call_json = mock_post.call_args[1]["json"]
        assert call_json["input"] == "hello world"


def test_get_embedding_ollama_error():
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("rag_common.requests.post", return_value=mock_response):
        vec = get_embedding("hello world")
        assert vec is None


from rag_common import load_index, save_index, add_to_index


def test_load_index_missing_file():
    vecs, ids = load_index("/nonexistent/path.npz")
    assert vecs is None
    assert ids == []


def test_save_and_load_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.npz")
        vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        ids = ["hash_a", "hash_b"]
        save_index(vecs, ids, path)

        loaded_vecs, loaded_ids = load_index(path)
        np.testing.assert_array_almost_equal(loaded_vecs, vecs)
        assert loaded_ids == ids


def test_add_to_index_empty():
    new_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vecs, ids = add_to_index(None, [], new_vec, "hash_new")
    assert vecs.shape == (1, 3)
    assert ids == ["hash_new"]


def test_add_to_index_existing():
    existing_vecs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    existing_ids = ["hash_a"]
    new_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    vecs, ids = add_to_index(existing_vecs, existing_ids, new_vec, "hash_b")
    assert vecs.shape == (2, 3)
    assert ids == ["hash_a", "hash_b"]


def test_add_to_index_dedup():
    existing_vecs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    existing_ids = ["hash_a"]
    new_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    vecs, ids = add_to_index(existing_vecs, existing_ids, new_vec, "hash_a")
    # Should not add duplicate
    assert vecs.shape == (1, 3)
    assert ids == ["hash_a"]


from rag_common import remove_from_index


def test_remove_from_index_basic():
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ids = ["a", "b", "c"]
    new_vecs, new_ids = remove_from_index(vecs, ids, {"b"})
    assert new_vecs.shape == (2, 3)
    assert new_ids == ["a", "c"]
    np.testing.assert_array_equal(new_vecs[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(new_vecs[1], [0.0, 0.0, 1.0])


def test_remove_from_index_all():
    vecs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    ids = ["a"]
    new_vecs, new_ids = remove_from_index(vecs, ids, {"a"})
    assert new_vecs is None
    assert new_ids == []


def test_remove_from_index_none():
    new_vecs, new_ids = remove_from_index(None, [], {"a"})
    assert new_vecs is None
    assert new_ids == []


def test_remove_from_index_no_match():
    vecs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    ids = ["a"]
    new_vecs, new_ids = remove_from_index(vecs, ids, {"x"})
    assert new_vecs.shape == (1, 3)
    assert new_ids == ["a"]


from rag_common import compute_file_checksum


def test_compute_file_checksum():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        f.flush()
        path = f.name
    try:
        h = compute_file_checksum(path)
        # "hello world" has no newlines, so normalization doesn't change it
        assert h == hashlib.sha256(b"hello world").hexdigest()
        assert len(h) == 64
    finally:
        os.unlink(path)


def test_compute_file_checksum_changes_on_content():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("version 1")
        f.flush()
        path = f.name
    try:
        h1 = compute_file_checksum(path)
        with open(path, "w") as f:
            f.write("version 2")
        h2 = compute_file_checksum(path)
        assert h1 != h2
    finally:
        os.unlink(path)


from rag_common import load_code_meta, save_code_meta


def test_load_code_meta_missing():
    meta = load_code_meta("/nonexistent/path.json")
    assert meta == {"checksums": {}, "chunks": {}}


def test_save_and_load_code_meta():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta.json")
        meta = {
            "checksums": {"src/Foo.cs": "abc123"},
            "chunks": {
                "src/Foo.cs::0": {"file": "src/Foo.cs", "start_line": 1, "end_line": 200}
            },
        }
        save_code_meta(meta, path)
        loaded = load_code_meta(path)
        assert loaded == meta


from rag_common import search


def test_search_returns_top_k():
    vecs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7, 0.7, 0.0],  # not normalized, but fine for test
    ], dtype=np.float32)
    ids = ["a", "b", "c"]
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = search(query, vecs, ids, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "a"  # exact match
    assert results[1][0] == "c"  # partial match


def test_search_empty_index():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = search(query, None, [], top_k=5)
    assert results == []
