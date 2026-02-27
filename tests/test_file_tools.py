"""Tests for file tools infrastructure."""

import os
import tempfile
import shutil
import pytest
from pathlib import Path

from holonpolis.infrastructure.storage.file_tools import FileTools, create_file_tools, ToolResult
from holonpolis.infrastructure.storage.file_io import FileIOError, PathGuardError
from holonpolis.infrastructure.storage.unified_apply import (
    EditType,
    FileOperation,
    parse_all_operations,
    apply_all_operations,
    apply_operation,
)


class TestFileToolsBasics:
    """Test basic FileTools functionality."""

    def test_create_file_tools(self):
        """Test creating FileTools instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = create_file_tools(tmpdir)
            assert tools.workspace == os.path.abspath(tmpdir)

    def test_file_tools_invalid_workspace(self):
        """Test FileTools with non-existent workspace."""
        with pytest.raises(ValueError, match="Workspace does not exist"):
            FileTools("/nonexistent/path/12345")

    def test_workspace_resolution(self):
        """Test workspace path resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a marker
            os.makedirs(os.path.join(tmpdir, "docs"))
            tools = FileTools(tmpdir)
            assert os.path.abspath(tmpdir) == tools.workspace


class TestReadOperations:
    """Test read operations."""

    def test_read_file(self):
        """Test reading a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("hello world")

            result = tools.read("workspace/test.txt")
            assert result.success is True
            assert result.data["content"] == "hello world"
            assert result.data["path"] == "workspace/test.txt"

    def test_read_file_not_found(self):
        """Test reading non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            result = tools.read("workspace/nonexistent.txt")
            assert result.success is False
            assert "not found" in result.error.lower()

    def test_read_file_with_offset_limit(self):
        """Test reading with offset and limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("line1\nline2\nline3\nline4\nline5")

            result = tools.read("workspace/test.txt", offset=1, limit=2)
            assert result.success is True
            assert result.data["content"] == "line2\nline3"
            assert result.data["offset"] == 1
            assert result.data["limit"] == 2

    def test_read_head(self):
        """Test reading file head."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            test_file = os.path.join(tmpdir, "test.txt")
            content = "x" * 1000
            with open(test_file, "w") as f:
                f.write(content)

            result = tools.read_head("workspace/test.txt", max_chars=100)
            assert result.success is True
            assert len(result.data["content"]) == 100

    def test_read_tail(self):
        """Test reading file tail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            test_file = os.path.join(tmpdir, "test.txt")
            lines = [f"line{i}" for i in range(20)]
            with open(test_file, "w") as f:
                f.write("\n".join(lines))

            result = tools.read_tail("workspace/test.txt", max_lines=5)
            assert result.success is True
            assert result.data["lines_read"] == 5
            assert "line19" in result.data["content"]


class TestWriteOperations:
    """Test write operations."""

    def test_write_file(self):
        """Test writing a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            result = tools.write("workspace/test.txt", "hello world")
            assert result.success is True

            test_file = os.path.join(tmpdir, "test.txt")
            assert os.path.exists(test_file)
            with open(test_file) as f:
                assert f.read() == "hello world"

    def test_write_file_creates_directories(self):
        """Test that write creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            result = tools.write("workspace/subdir/nested/file.txt", "content")
            assert result.success is True

            test_file = os.path.join(tmpdir, "subdir", "nested", "file.txt")
            assert os.path.exists(test_file)

    def test_write_json_file(self):
        """Test writing JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            data = {"key": "value", "number": 42}
            result = tools.write_json("workspace/data.json", data)
            assert result.success is True

            import json
            test_file = os.path.join(tmpdir, "data.json")
            with open(test_file) as f:
                loaded = json.load(f)
                assert loaded == data


class TestDirectoryOperations:
    """Test directory operations."""

    def test_list_directory(self):
        """Test listing directory contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            # Create test files
            os.makedirs(os.path.join(tmpdir, "subdir"))
            with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
                f.write("content")

            result = tools.list("workspace/")
            assert result.success is True
            assert result.data["total"] == 2

            names = [e["name"] for e in result.data["entries"]]
            assert "file1.txt" in names
            assert "subdir" in names

    def test_list_recursive(self):
        """Test recursive directory listing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            os.makedirs(os.path.join(tmpdir, "a", "b", "c"))
            with open(os.path.join(tmpdir, "a", "file.txt"), "w") as f:
                f.write("content")

            result = tools.list("workspace/", recursive=True)
            assert result.success is True
            # Should find a/, a/b/, a/b/c/, a/file.txt
            assert result.data["total"] >= 4

    def test_list_with_pattern(self):
        """Test listing with pattern filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            with open(os.path.join(tmpdir, "file.py"), "w") as f:
                f.write("python")
            with open(os.path.join(tmpdir, "file.txt"), "w") as f:
                f.write("text")

            result = tools.list("workspace/", pattern="*.py")
            assert result.success is True
            assert result.data["total"] == 1
            assert result.data["entries"][0]["name"] == "file.py"

    def test_mkdir(self):
        """Test creating directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            result = tools.mkdir("workspace/newdir")
            assert result.success is True
            assert os.path.isdir(os.path.join(tmpdir, "newdir"))

    def test_mkdir_nested(self):
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            result = tools.mkdir("workspace/a/b/c", parents=True)
            assert result.success is True
            assert os.path.isdir(os.path.join(tmpdir, "a", "b", "c"))

    def test_rmdir(self):
        """Test removing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            os.mkdir(os.path.join(tmpdir, "emptydir"))

            result = tools.rmdir("workspace/emptydir")
            assert result.success is True
            assert not os.path.exists(os.path.join(tmpdir, "emptydir"))

    def test_rmdir_recursive(self):
        """Test removing directory recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            os.makedirs(os.path.join(tmpdir, "dir", "subdir"))
            with open(os.path.join(tmpdir, "dir", "file.txt"), "w") as f:
                f.write("content")

            result = tools.rmdir("workspace/dir", recursive=True)
            assert result.success is True
            assert not os.path.exists(os.path.join(tmpdir, "dir"))


class TestSearchOperations:
    """Test search operations."""

    def test_search_files(self):
        """Test searching in files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            with open(os.path.join(tmpdir, "file1.py"), "w") as f:
                f.write("def hello(): pass")
            with open(os.path.join(tmpdir, "file2.py"), "w") as f:
                f.write("def world(): pass")
            with open(os.path.join(tmpdir, "file.txt"), "w") as f:
                f.write("no match here")

            result = tools.search("def ", pattern="*.py")
            assert result.success is True
            assert result.data["total"] == 2

    def test_search_with_max_results(self):
        """Test search with max results limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            for i in range(10):
                with open(os.path.join(tmpdir, f"file{i}.py"), "w") as f:
                    f.write(f"def func{i}(): pass")

            result = tools.search("def ", pattern="*.py", max_results=5)
            assert result.success is True
            assert result.data["total"] == 5
            assert result.data["truncated"] is True


class TestDeleteOperations:
    """Test delete operations."""

    def test_delete_file(self):
        """Test deleting a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("content")

            result = tools.delete("workspace/test.txt")
            assert result.success is True
            assert result.data["deleted"] is True
            assert not os.path.exists(test_file)

    def test_delete_nonexistent_file(self):
        """Test deleting non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            result = tools.delete("workspace/nonexistent.txt")
            assert result.success is True
            assert result.data["deleted"] is False

    def test_delete_must_exist(self):
        """Test deleting with must_exist flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            result = tools.delete("workspace/nonexistent.txt", must_exist=True)
            assert result.success is False
            assert "not found" in result.error.lower()


class TestPatchOperations:
    """Test patch/apply operations."""

    def test_parse_patch_search_replace(self):
        """Test parsing SEARCH/REPLACE patch."""
        patch_text = """
PATCH_FILE: test.py
<<<<<<< SEARCH
def old():
    pass
=======
def new():
    return True
>>>>>>> REPLACE
END PATCH_FILE
"""
        operations = parse_all_operations(patch_text)
        assert len(operations) == 1
        assert operations[0].edit_type == EditType.SEARCH_REPLACE
        assert operations[0].path == "test.py"
        assert "def old()" in operations[0].search
        assert "def new()" in operations[0].replace

    def test_parse_patch_create_file(self):
        """Test parsing CREATE file patch."""
        patch_text = """
CREATE: new_file.py
print("hello")
END CREATE
"""
        operations = parse_all_operations(patch_text)
        assert len(operations) == 1
        assert operations[0].edit_type == EditType.CREATE
        assert operations[0].path == "new_file.py"
        assert 'print("hello")' in operations[0].replace

    def test_parse_patch_delete_file(self):
        """Test parsing DELETE file patch."""
        patch_text = """
DELETE_FILE: old_file.py
"""
        operations = parse_all_operations(patch_text)
        assert len(operations) == 1
        assert operations[0].edit_type == EditType.DELETE
        assert operations[0].path == "old_file.py"

    def test_apply_patch_search_replace(self):
        """Test applying SEARCH/REPLACE patch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("def old():\n    pass\n")

            patch_text = """
PATCH_FILE: test.py
<<<<<<< SEARCH
def old():
    pass
=======
def new():
    return True
>>>>>>> REPLACE
END PATCH_FILE
"""
            result = apply_all_operations(patch_text, tmpdir)
            assert result.success is True
            assert "test.py" in result.changed_files

            with open(test_file) as f:
                content = f.read()
                assert "def new():" in content

    def test_apply_patch_create_file(self):
        """Test applying CREATE file patch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_text = """
CREATE: new_module.py
def hello():
    return "world"
END CREATE
"""
            result = apply_all_operations(patch_text, tmpdir)
            assert result.success is True

            test_file = os.path.join(tmpdir, "new_module.py")
            assert os.path.exists(test_file)
            with open(test_file) as f:
                assert "def hello():" in f.read()

    def test_apply_patch_delete_file(self):
        """Test applying DELETE file patch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "to_delete.py")
            with open(test_file, "w") as f:
                f.write("content")

            patch_text = """
DELETE_FILE: to_delete.py
"""
            result = apply_all_operations(patch_text, tmpdir)
            assert result.success is True
            assert "to_delete.py" in result.changed_files
            assert not os.path.exists(test_file)

    def test_apply_patch_with_file_tools(self):
        """Test applying patch via FileTools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("old content\n")

            patch_text = """
PATCH_FILE: test.py
<<<<<<< SEARCH
old content
=======
new content
>>>>>>> REPLACE
END PATCH_FILE
"""
            result = tools.apply_patch(patch_text)
            assert result.success is True
            assert "test.py" in result.data["changed_files"]

    def test_parse_patch_dry_run(self):
        """Test dry run patch parsing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            patch_text = """
CREATE: file1.py
content1
END CREATE

CREATE: file2.py
content2
END CREATE
"""
            result = tools.apply_patch(patch_text, dry_run=True)
            assert result.success is True
            assert result.data["dry_run"] is True
            assert result.data["operations_found"] == 2


class TestPathSecurity:
    """Test path traversal security."""

    def test_path_traversal_blocked(self):
        """Test that path traversal is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            # Create a file outside workspace
            outside_file = os.path.join(os.path.dirname(tmpdir), "outside.txt")
            with open(outside_file, "w") as f:
                f.write("secret")

            # Try to read it via path traversal
            traversal_path = "../" + os.path.basename(os.path.dirname(tmpdir)) + "/outside.txt"
            result = tools.read(f"workspace/{traversal_path}")
            # Should fail with path error
            assert result.success is False

    def test_absolute_path_blocked(self):
        """Test that absolute paths outside workspace are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            outside_file = "/etc/passwd" if os.name != "nt" else "C:/Windows/system.ini"

            result = tools.read(f"workspace/{outside_file}")
            assert result.success is False


class TestUtilityMethods:
    """Test utility methods."""

    def test_exists(self):
        """Test exists method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            with open(os.path.join(tmpdir, "exists.txt"), "w") as f:
                f.write("content")

            assert tools.exists("workspace/exists.txt") is True
            assert tools.exists("workspace/nonexistent.txt") is False

    def test_is_file(self):
        """Test is_file method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            with open(os.path.join(tmpdir, "file.txt"), "w") as f:
                f.write("content")
            os.mkdir(os.path.join(tmpdir, "dir"))

            assert tools.is_file("workspace/file.txt") is True
            assert tools.is_file("workspace/dir") is False
            assert tools.is_file("workspace/nonexistent") is False

    def test_is_dir(self):
        """Test is_dir method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = FileTools(tmpdir)
            with open(os.path.join(tmpdir, "file.txt"), "w") as f:
                f.write("content")
            os.mkdir(os.path.join(tmpdir, "dir"))

            assert tools.is_dir("workspace/dir") is True
            assert tools.is_dir("workspace/file.txt") is False
            assert tools.is_dir("workspace/nonexistent") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
