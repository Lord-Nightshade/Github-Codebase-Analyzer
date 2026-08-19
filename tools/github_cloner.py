import os
import shutil
import tempfile
import git

from config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_KB,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def clone_and_parse_repo(repo_url: str) -> list[dict]:
    """
    Clones a public GitHub repository, traverses supported code files,
    chunks content, and returns a list of chunk dictionaries.
    
    Each dictionary contains:
    - 'content': str (the code block)
    - 'file_path': str (relative path inside the repo)
    - 'chunk_index': int
    """
    temp_dir = tempfile.mkdtemp(prefix="repo_clone_")
    code_chunks = []

    try:
        print(f"Clones repository: {repo_url} (shallow clone depth=1)...")
        git.Repo.clone_from(repo_url, temp_dir, depth=1)

        for root, dirs, files in os.walk(temp_dir):
            # Exclude hidden directories (like .git, .github)
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                
                # Also allow exact file matches like 'Dockerfile'
                if ext in ALLOWED_EXTENSIONS or file in ALLOWED_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, temp_dir)

                    # Skip files exceeding size limits (e.g., minified JS or large generated files)
                    file_size_kb = os.path.getsize(full_path) / 1024
                    if file_size_kb > MAX_FILE_SIZE_KB:
                        print(f"Skipping oversized file ({file_size_kb:.1f} KB): {rel_path}")
                        continue

                    # Read file content safely
                    content = _read_file_safe(full_path)
                    if not content or not content.strip():
                        continue

                    # Split file content into chunks
                    file_chunks = _chunk_text(content, rel_path)
                    code_chunks.extend(file_chunks)

        print(f"Parsing complete. Generated {len(code_chunks)} chunks across repository.")
        return code_chunks

    except Exception as e:
        print(f"Error cloning or parsing repository {repo_url}: {e}")
        raise e

    finally:
        # Clean up temporary disk files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def _read_file_safe(file_path: str) -> str:
    """Reads text content handling UTF-8 and fallback encodings."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception:
            return ""
    return ""


def _chunk_text(text: str, file_path: str) -> list[dict]:
    """Splits source code text into overlapping character chunks."""
    chunks = []
    start = 0
    text_length = len(text)
    chunk_idx = 0

    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)
        chunk_content = text[start:end]

        # Format chunk to include file header context for the LLM
        formatted_content = f"// File: {file_path}\n{chunk_content}"

        chunks.append({
            "content": formatted_content,
            "file_path": file_path,
            "chunk_index": chunk_idx
        })

        chunk_idx += 1
        if end == text_length:
            break
        
        start += (CHUNK_SIZE - CHUNK_OVERLAP)

    return chunks