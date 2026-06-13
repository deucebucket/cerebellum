"""
build_coder_corpus.py -- Assemble coder_corpus_v2.txt for cerebellum-brainloop-coder-python.

Sources:
  1. python_stdlib_13k.txt   -- raw symbol docs (PRIMARY, full)
  2. QA pairs: every 4th doc (25% of docs) in chat Q&A format
  3. Completion pairs: every 4th doc offset by 2 (another 25%) in code-completion shape:
       "{signature line}\n    \"\"\"{first doc content line}\"\"\"\n"
  4. code_corpus_full.txt + code_train_focused.txt  -- code examples (full)
  5. wiki_code_combined.raw  -- general-knowledge regularizer (first 6 MB, up from 4 MB)

Interleave: cycle of (20 stdlib docs, 10 QA pairs, 10 completion pairs, 1 code chunk, 1 wiki chunk)
so no single source dominates any contiguous region of the file.

Output: coder_corpus_v2.txt (same directory as this script)
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STDLIB_PATH   = os.path.join(SCRIPT_DIR, "python_stdlib_13k.txt")
CODE_FULL     = os.path.join(SCRIPT_DIR, "code_corpus_full.txt")
CODE_FOCUSED  = os.path.join(SCRIPT_DIR, "code_train_focused.txt")
WIKI_PATH     = os.path.join(SCRIPT_DIR, "wiki_code_combined.raw")
OUTPUT_PATH   = os.path.join(SCRIPT_DIR, "coder_corpus_v2.txt")

WIKI_LIMIT_BYTES = 6 * 1024 * 1024  # 6 MB (increased from 4 MB for stronger regularizer)
CHUNK_SIZE       = 4096              # bytes per code/wiki chunk
BATCH_SIZE       = 20                # stdlib docs per batch in the cycle
QA_BATCH         = 10                # QA pairs per batch (25% of docs = every 4th)
COMP_BATCH       = 10                # completion pairs per batch (another 25%, offset by 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_stdlib_docs(text: str) -> list[str]:
    """Split on '# symbol' section headers; each entry starts with the header."""
    parts = []
    current = []
    for line in text.splitlines(keepends=True):
        if line.startswith("# ") and current:
            parts.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("".join(current))
    return parts


def doc_to_qa(doc: str) -> str:
    """Convert a stdlib doc block to a Q&A pair."""
    lines = doc.strip().splitlines()
    if not lines:
        return doc
    # First line is '# symbol.name'; strip the '# '
    symbol = lines[0].lstrip("# ").strip()
    body = "\n".join(lines[1:]).strip()
    return (
        f"Question: How do I use {symbol} in Python?\n"
        f"Answer: {body}\n\n"
    )


def doc_to_completion(doc: str) -> str:
    """Convert a stdlib doc block to a code-completion stub.

    Shape:  <signature line>
                \"\"\"<first content line>\"\"\"

    The signature is derived deterministically from the symbol name on the
    first line of the doc:
      - If the symbol contains '.' (e.g. 'os.path.join') the last component
        becomes the function name and the module becomes the apparent context.
      - A bare name (e.g. 'print') is used directly.
    This keeps the transformation simple and avoids hallucinating signatures
    we don't have ground truth for.
    """
    lines = doc.strip().splitlines()
    if not lines:
        return doc
    symbol = lines[0].lstrip("# ").strip()
    # First non-empty content line as the docstring body
    first_content = ""
    for line in lines[1:]:
        stripped = line.strip()
        if stripped:
            first_content = stripped
            break
    if not first_content:
        first_content = symbol

    # Build a minimal signature: def <last_component>(*args, **kwargs):
    last_component = symbol.split(".")[-1] if "." in symbol else symbol
    # Sanitize: replace any chars that aren't valid Python identifiers
    safe_name = "".join(c if (c.isalnum() or c == "_") else "_" for c in last_component)
    if not safe_name or safe_name[0].isdigit():
        safe_name = "f_" + safe_name

    signature = f"def {safe_name}(*args, **kwargs):"
    return f'{signature}\n    """{first_content}"""\n'


def byte_chunks(text: str, chunk_size: int) -> list[str]:
    """Split text into ~chunk_size byte chunks on newline boundaries."""
    encoded = text.encode("utf-8")
    chunks = []
    pos = 0
    total = len(encoded)
    while pos < total:
        end = min(pos + chunk_size, total)
        # Walk back to a newline to avoid splitting mid-character
        if end < total:
            nl = encoded.rfind(b"\n", pos, end)
            if nl > pos:
                end = nl + 1
        chunks.append(encoded[pos:end].decode("utf-8", errors="replace"))
        pos = end
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Reading source files...", flush=True)

    with open(STDLIB_PATH, "r", encoding="utf-8") as f:
        stdlib_text = f.read()

    with open(CODE_FULL, "r", encoding="utf-8") as f:
        code_full_text = f.read()

    with open(CODE_FOCUSED, "r", encoding="utf-8") as f:
        code_focused_text = f.read()

    # wiki: read only first WIKI_LIMIT_BYTES
    with open(WIKI_PATH, "rb") as f:
        wiki_raw = f.read(WIKI_LIMIT_BYTES)
    wiki_text = wiki_raw.decode("utf-8", errors="replace")

    # Parse stdlib docs (full)
    stdlib_docs = split_stdlib_docs(stdlib_text)

    # QA pairs: every 4th doc starting at index 0 (25% of docs)
    qa_docs = [doc_to_qa(stdlib_docs[i]) for i in range(0, len(stdlib_docs), 4)]

    # Completion pairs: every 4th doc starting at index 2 (another 25%, non-overlapping)
    comp_docs = [doc_to_completion(stdlib_docs[i]) for i in range(2, len(stdlib_docs), 4)]

    code_chunks = byte_chunks(code_full_text + "\n" + code_focused_text, CHUNK_SIZE)
    wiki_chunks = byte_chunks(wiki_text, CHUNK_SIZE)

    print(f"  stdlib docs:         {len(stdlib_docs)}")
    print(f"  QA pairs (every 4th, offset 0): {len(qa_docs)}")
    print(f"  completion pairs (every 4th, offset 2): {len(comp_docs)}")
    print(f"  code chunks:         {len(code_chunks)}")
    print(f"  wiki chunks:         {len(wiki_chunks)}")

    # Interleave: cycle (20 stdlib docs, 10 QA pairs, 10 completion pairs,
    #                    1 code chunk, 1 wiki chunk)
    out_parts = []

    stdlib_i = 0
    qa_i     = 0
    comp_i   = 0
    code_i   = 0
    wiki_i   = 0

    stdlib_bytes = 0
    qa_bytes     = 0
    comp_bytes   = 0
    code_bytes   = 0
    wiki_bytes   = 0

    while (stdlib_i < len(stdlib_docs) or qa_i < len(qa_docs)
           or comp_i < len(comp_docs)
           or code_i < len(code_chunks) or wiki_i < len(wiki_chunks)):

        # 20 stdlib docs
        for _ in range(BATCH_SIZE):
            if stdlib_i >= len(stdlib_docs):
                break
            block = stdlib_docs[stdlib_i]
            out_parts.append(block)
            stdlib_bytes += len(block.encode("utf-8"))
            stdlib_i += 1

        # 10 QA pairs (25% of docs, every 4th)
        for _ in range(QA_BATCH):
            if qa_i >= len(qa_docs):
                break
            block = qa_docs[qa_i]
            out_parts.append(block)
            qa_bytes += len(block.encode("utf-8"))
            qa_i += 1

        # 10 completion pairs (another 25%, offset by 2)
        for _ in range(COMP_BATCH):
            if comp_i >= len(comp_docs):
                break
            block = comp_docs[comp_i]
            out_parts.append(block)
            comp_bytes += len(block.encode("utf-8"))
            comp_i += 1

        # 1 code chunk
        if code_i < len(code_chunks):
            block = code_chunks[code_i]
            out_parts.append(block)
            code_bytes += len(block.encode("utf-8"))
            code_i += 1

        # 1 wiki chunk
        if wiki_i < len(wiki_chunks):
            block = wiki_chunks[wiki_i]
            out_parts.append(block)
            wiki_bytes += len(block.encode("utf-8"))
            wiki_i += 1

    corpus = "\n".join(out_parts)
    total_bytes = len(corpus.encode("utf-8"))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"\ncoder_corpus_v2.txt written to {OUTPUT_PATH}")
    print(f"  stdlib docs bytes:        {stdlib_bytes:>10,}")
    print(f"  QA pairs bytes:           {qa_bytes:>10,}")
    print(f"  completion pairs bytes:   {comp_bytes:>10,}")
    print(f"  code bytes:               {code_bytes:>10,}")
    print(f"  wiki bytes:               {wiki_bytes:>10,}")
    print(f"  TOTAL bytes:              {total_bytes:>10,}  ({total_bytes/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
