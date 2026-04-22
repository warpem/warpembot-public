import os
import tempfile

import numpy as np
from unittest.mock import patch, MagicMock

from rag_common import load_index, load_code_meta, code_index_path, code_meta_path
from rag_index import chunk_file, walk_repo_files, diff_checksums, cmd_update_repos


def test_chunk_file_small():
    """A file shorter than chunk_size produces one chunk."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("line1\nline2\nline3\n")
        path = f.name
    try:
        chunks = chunk_file(path, "src/small.py")
        assert len(chunks) == 1
        chunk_id, text, start, end = chunks[0]
        assert chunk_id == "src/small.py::0"
        assert text.startswith("# File: src/small.py\n")
        assert "line1" in text
        assert start == 1
        assert end == 3
    finally:
        os.unlink(path)


def test_chunk_file_large():
    """A 450-line file with chunk_size=200, overlap=50 produces 3 chunks."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cs", delete=False) as f:
        for i in range(450):
            f.write(f"line {i}\n")
        path = f.name
    try:
        chunks = chunk_file(path, "src/Big.cs", chunk_size=200, overlap=50)
        assert len(chunks) == 3
        # Chunk 0: lines 1-200
        assert chunks[0][0] == "src/Big.cs::0"
        assert chunks[0][2] == 1
        assert chunks[0][3] == 200
        # Chunk 1: lines 151-350
        assert chunks[1][0] == "src/Big.cs::1"
        assert chunks[1][2] == 151
        assert chunks[1][3] == 350
        # Chunk 2: lines 301-450
        assert chunks[2][0] == "src/Big.cs::2"
        assert chunks[2][2] == 301
        assert chunks[2][3] == 450
    finally:
        os.unlink(path)


def test_chunk_file_empty():
    """An empty file produces no chunks."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        path = f.name
    try:
        chunks = chunk_file(path, "src/empty.py")
        assert chunks == []
    finally:
        os.unlink(path)


def test_chunk_file_prefix():
    """Each chunk text starts with the file path comment."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("x = 1\n")
        path = f.name
    try:
        chunks = chunk_file(path, "lib/util.py")
        assert chunks[0][1].startswith("# File: lib/util.py\n")
    finally:
        os.unlink(path)


def test_walk_repo_files_includes_extensions(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.cs").write_text("class Foo {}")
    (tmp_path / "src" / "helper.py").write_text("def foo(): pass")
    (tmp_path / "src" / "data.bin").write_text("binary")  # should be excluded
    result = walk_repo_files(str(tmp_path))
    assert "src/main.cs" in result
    assert "src/helper.py" in result
    assert "src/data.bin" not in result


def test_walk_repo_files_excludes_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.cs").write_text("class Foo {}")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "output.cs").write_text("nope")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.py").write_text("nope")
    result = walk_repo_files(str(tmp_path))
    assert "src/main.cs" in result
    assert not any("bin/" in r for r in result)
    assert not any(".git/" in r for r in result)


def test_walk_repo_files_excludes_generated(tmp_path):
    (tmp_path / "Foo.generated.cs").write_text("auto")
    (tmp_path / "Bar.cs").write_text("manual")
    result = walk_repo_files(str(tmp_path))
    assert "Bar.cs" in result
    assert "Foo.generated.cs" not in result


def test_walk_repo_files_returns_sorted(tmp_path):
    (tmp_path / "z.py").write_text("")
    (tmp_path / "a.py").write_text("")
    (tmp_path / "m.py").write_text("")
    result = walk_repo_files(str(tmp_path))
    assert result == sorted(result)


def test_diff_checksums_all_new():
    current = {"a.py": "hash_a", "b.py": "hash_b"}
    stored = {}
    new, changed, deleted = diff_checksums(current, stored)
    assert sorted(new) == ["a.py", "b.py"]
    assert changed == []
    assert deleted == []


def test_diff_checksums_changed():
    current = {"a.py": "hash_new"}
    stored = {"a.py": "hash_old"}
    new, changed, deleted = diff_checksums(current, stored)
    assert new == []
    assert changed == ["a.py"]
    assert deleted == []


def test_diff_checksums_deleted():
    current = {}
    stored = {"a.py": "hash_a"}
    new, changed, deleted = diff_checksums(current, stored)
    assert new == []
    assert changed == []
    assert deleted == ["a.py"]


def test_diff_checksums_mixed():
    current = {"a.py": "hash_a", "b.py": "hash_b_new", "c.py": "hash_c"}
    stored = {"b.py": "hash_b_old", "d.py": "hash_d"}
    new, changed, deleted = diff_checksums(current, stored)
    assert sorted(new) == ["a.py", "c.py"]
    assert changed == ["b.py"]
    assert deleted == ["d.py"]


def test_diff_checksums_no_changes():
    checksums = {"a.py": "hash_a", "b.py": "hash_b"}
    new, changed, deleted = diff_checksums(checksums, checksums)
    assert new == []
    assert changed == []
    assert deleted == []


def _make_embed_mock():
    """Create a mock that returns the right number of embeddings for batch requests."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    def json_side_effect(*args, **kwargs):
        # Return as many embeddings as inputs were sent
        call_json = mock_post.call_args
        if call_json:
            input_val = call_json[1].get("json", {}).get("input", "")
            count = len(input_val) if isinstance(input_val, list) else 1
        else:
            count = 1
        return {"embeddings": [[0.1, 0.2, 0.3]] * count}

    mock_response.json = json_side_effect
    mock_post = MagicMock(return_value=mock_response)
    return mock_post


def _patch_repos(tmp_path, repo_names=None):
    """Helper: returns a context manager that patches repo paths for testing."""
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        repos_dir = tmp_path / "repos"
        repos_dir.mkdir(exist_ok=True)
        with patch("rag_index.REPOS_DIR", str(repos_dir)), \
             patch("rag_index.REPO_NAMES", repo_names or ["testrepo"]), \
             patch("rag_index.code_index_path", lambda r: str(tmp_path / f"code_index_{r}.npz")), \
             patch("rag_index.code_meta_path", lambda r: str(tmp_path / f"code_meta_{r}.json")):
            yield repos_dir
    return ctx()


def test_cmd_update_repos_initial_build(tmp_path):
    """First run should embed all files in the repo."""
    mock_post = _make_embed_mock()

    with patch("rag_common.requests.post", mock_post), \
         patch("rag_index.requests.get"), \
         _patch_repos(tmp_path) as repos_dir:
        repo = repos_dir / "testrepo"
        repo.mkdir()
        (repo / "main.py").write_text("def hello():\n    return 'world'\n")
        cmd_update_repos()

    # Verify index was created
    idx_path = str(tmp_path / "code_index_testrepo.npz")
    meta_path = str(tmp_path / "code_meta_testrepo.json")
    vecs, ids = load_index(idx_path)
    meta = load_code_meta(meta_path)

    assert vecs is not None
    assert len(ids) == 1
    assert ids[0] == "main.py::0"
    assert "main.py" in meta["checksums"]
    assert "main.py::0" in meta["chunks"]


def test_cmd_update_repos_incremental(tmp_path):
    """Second run with no changes should embed nothing."""
    mock_post = _make_embed_mock()

    with patch("rag_common.requests.post", mock_post), \
         patch("rag_index.requests.get"), \
         _patch_repos(tmp_path) as repos_dir:
        repo = repos_dir / "testrepo"
        repo.mkdir()
        (repo / "main.py").write_text("def hello():\n    return 'world'\n")
        cmd_update_repos()

        # Second run — no changes
        mock_post.reset_mock()
        cmd_update_repos()
        assert mock_post.call_count == 0  # No embeddings computed


def test_cmd_update_repos_detects_change(tmp_path):
    """Changing a file should re-embed it."""
    mock_post = _make_embed_mock()

    with patch("rag_common.requests.post", mock_post), \
         patch("rag_index.requests.get"), \
         _patch_repos(tmp_path) as repos_dir:
        repo = repos_dir / "testrepo"
        repo.mkdir()
        (repo / "main.py").write_text("version 1\n")
        cmd_update_repos()

        # Change the file
        (repo / "main.py").write_text("version 2\n")
        mock_post.reset_mock()
        cmd_update_repos()
        assert mock_post.call_count == 1  # Re-embedded the changed file


def test_cmd_update_repos_detects_deletion(tmp_path):
    """Deleting a file should remove its chunks from the index."""
    mock_post = _make_embed_mock()

    with patch("rag_common.requests.post", mock_post), \
         patch("rag_index.requests.get"), \
         _patch_repos(tmp_path) as repos_dir:
        repo = repos_dir / "testrepo"
        repo.mkdir()
        (repo / "main.py").write_text("version 1\n")
        (repo / "helper.py").write_text("def help(): pass\n")
        cmd_update_repos()

        # Verify both files indexed
        idx_path = str(tmp_path / "code_index_testrepo.npz")
        vecs, ids = load_index(idx_path)
        assert len(ids) == 2

        # Delete one file
        (repo / "helper.py").unlink()
        cmd_update_repos()

        # Verify only main.py remains
        vecs, ids = load_index(idx_path)
        assert len(ids) == 1
        assert ids[0] == "main.py::0"
        meta = load_code_meta(str(tmp_path / "code_meta_testrepo.json"))
        assert "helper.py" not in meta["checksums"]
        assert not any(k.startswith("helper.py::") for k in meta["chunks"])
