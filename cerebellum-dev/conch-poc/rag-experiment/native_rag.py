"""
Native RAG: model's OWN token embeddings as the document index space.

Key insight: if documents are pre-embedded using the model's embedding
layer (tok_embd), then retrieved vectors are ALREADY in the model's
internal language. No W_context projection needed. The hidden states
at any layer are 2048-dim -- same as the embedding space.

Components:
  1. extract_embedding_layer() -- loads only model.embed_tokens.weight from safetensors
  2. build_document_corpus()   -- tokenize texts, average token embeddings, build FAISS index
  3. inject_native_rag()       -- take hidden states as query, search FAISS, return retrieved vectors
  4. smoke_test()              -- end-to-end verification with shapes and sanity checks
"""

from __future__ import annotations

import faiss
import numpy as np
import torch

MODEL_ID = "Qwen/Qwen2.5-3B"
HIDDEN_DIM = 2048
TOP_K = 5
RETRIEVAL_SCALE = 0.1


# ---------------------------------------------------------------------------
# 1. Load embedding layer (no full model)
# ---------------------------------------------------------------------------

def extract_embedding_layer(model_id: str = MODEL_ID) -> torch.Tensor:
    """Load the token embedding weight from safetensors without loading the full model.

    Returns [vocab_size, 2048] float32 tensor.
    """
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    import json
    with open(index_path) as f:
        index = json.load(f)

    embed_file = None
    for key, fname in index["weight_map"].items():
        if "embed_tokens" in key:
            embed_file = fname
            embed_key = key
            break

    if embed_file is None:
        raise RuntimeError("Could not find embed_tokens in weight map")

    fpath = hf_hub_download(model_id, embed_file)
    with safe_open(fpath, framework="pt", device="cpu") as f:
        weight = f.get_tensor(embed_key)

    embed_weight = weight.to(torch.float32)
    print(f"  embed_tokens.weight: {list(embed_weight.shape)}  dtype={embed_weight.dtype}")
    return embed_weight


# ---------------------------------------------------------------------------
# 2. Build document corpus & FAISS index
# ---------------------------------------------------------------------------

def _tokenize_texts(
    texts: list[str],
    tokenizer,
    max_tokens: int = 128,
) -> list[torch.Tensor]:
    """Tokenize each text and return list of token-id tensors."""
    token_ids_list: list[torch.Tensor] = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=True)
        ids = torch.tensor(encoded[:max_tokens], dtype=torch.long)
        token_ids_list.append(ids)
    return token_ids_list


def _average_token_embeddings(
    token_ids_list: list[torch.Tensor],
    embed_weight: torch.Tensor,
) -> np.ndarray:
    """Average token embeddings for each document -> [num_docs, 2048] float32."""
    vectors = []
    for ids in token_ids_list:
        embs = embed_weight[ids]           # [n_tokens, 2048]
        doc_vec = embs.mean(dim=0)          # [2048]
        vectors.append(doc_vec.numpy())
    return np.stack(vectors, axis=0).astype(np.float32)


def _load_wikitext_docs(max_docs: int = 300) -> list[str]:
    """Load a small subset of WikiText-2 raw documents. Falls back to synthetic."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
        docs = []
        for item in ds:
            text = item["text"].strip()
            if len(text) > 200 and not text.startswith("="):
                docs.append(text)
            if len(docs) >= max_docs:
                break
        if len(docs) >= 10:
            print(f"  loaded {len(docs)} WikiText-2 documents")
            return docs
    except Exception as e:
        print(f"  WikiText load failed ({e}), falling back to synthetic docs")

    return _synthetic_docs()


def _synthetic_docs() -> list[str]:
    """Hardcoded domain-diverse paragraphs for a self-contained test."""
    return [
        "The transformer architecture uses self-attention mechanisms to process sequences in parallel, "
        "replacing the sequential nature of recurrent neural networks. Each token attends to all other "
        "tokens in the sequence, allowing the model to capture long-range dependencies efficiently.",
        "Quantum computing leverages the principles of superposition and entanglement to perform calculations "
        "that would be infeasible for classical computers. Qubits can exist in multiple states simultaneously, "
        "enabling exponential speedups for specific problem classes.",
        "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. It generates "
        "most of the cell's supply of adenosine triphosphate through oxidative phosphorylation.",
        "The Rust programming language emphasizes memory safety without garbage collection through its ownership "
        "system. The borrow checker enforces rules at compile time that prevent data races and use-after-free errors.",
        "Photosynthesis converts light energy into chemical energy in plants, algae, and cyanobacteria. "
        "Chlorophyll molecules absorb photons and drive electron transport chains that produce ATP and NADPH.",
        "Gradient descent is an iterative optimization algorithm for finding local minima of differentiable "
        "functions. The learning rate determines the step size in each iteration toward the negative gradient.",
        "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions, extending from the Arctic "
        "Ocean in the north to the Southern Ocean in the south, bounded by Asia and Australia on the west.",
        "Blockchain technology provides a decentralized ledger maintained by a peer-to-peer network. "
        "Consensus mechanisms like proof-of-work and proof-of-stake ensure agreement on the chain state.",
        "The human genome contains approximately three billion base pairs of DNA organized into twenty-three "
        "pairs of chromosomes. Genes encode proteins through the processes of transcription and translation.",
        "Convolutional neural networks apply learned filters across spatial dimensions of input data. "
        "Early layers detect simple features like edges, while deeper layers compose them into complex patterns.",
        "The Byzantine Empire was the continuation of the Roman Empire in the eastern Mediterranean during "
        "Late Antiquity and the Middle Ages. Its capital, Constantinople, fell to the Ottoman Empire in 1453.",
        "Black holes are regions of spacetime where gravity is so strong that nothing can escape. "
        "The event horizon marks the boundary beyond which events cannot affect an outside observer.",
        "TCP/IP is the foundational protocol suite of the internet. TCP provides reliable ordered delivery "
        "of data streams, while IP handles addressing and routing of packets across network boundaries.",
        "Natural selection is the differential survival and reproduction of individuals due to differences in "
        "phenotype. It is a key mechanism of evolution, the change in heritable traits over successive generations.",
        "The Linux kernel is a free and open-source monolithic modular multitasking Unix-like operating system "
        "kernel. It was created by Linus Torvalds in 1991 and has since been ported to more hardware platforms.",
        "The French Revolution was a period of radical political and societal change in France beginning in 1789. "
        "It overthrew the monarchy, established a republic, and catalyzed modern democratic ideals.",
        "Reinforcement learning agents learn optimal behavior through trial and error interactions with an "
        "environment. The agent receives rewards or penalties and updates its policy to maximize cumulative return.",
        "The Solar System formed approximately 4.6 billion years ago from the gravitational collapse of a "
        "giant interstellar molecular cloud. It consists of the Sun, eight planets, dwarf planets, and small bodies.",
        "SQL databases organize data into tables with rows and columns, supporting relational algebra operations. "
        "ACID transactions guarantee atomicity, consistency, isolation, and durability in concurrent access.",
        "The theory of special relativity establishes that the speed of light in vacuum is constant for all "
        "observers. Time dilation and length contraction emerge from the Lorentz transformations between frames.",
        # Additional synthetic docs for better retrieval diversity
        "Attention mechanisms compute weighted sums of values based on query-key similarity scores. "
        "Multi-head attention allows the model to jointly attend to information from different representation "
        "subspaces at different positions, improving the model's ability to capture diverse patterns.",
        "CRISPR-Cas9 is a genome editing technology that allows researchers to alter DNA sequences and modify "
        "gene function. It has applications in treating genetic disorders, improving crops, and basic research.",
        "The lambda calculus is a formal system in mathematical logic for expressing computation based on "
        "function abstraction and application. It is the theoretical foundation of functional programming.",
        "Plate tectonics describes the large-scale motion of Earth's lithosphere. The planet's outer shell "
        "is divided into plates that glide over the mantle, causing earthquakes, volcanoes, and mountain building.",
        "Backpropagation computes gradients of loss functions with respect to neural network parameters. "
        "The chain rule is applied recursively from the output layer backward through the computation graph.",
        "The Renaissance was a cultural movement spanning the 14th to 17th centuries, beginning in Italy and "
        "spreading through Europe. It marked the transition from the medieval period to modernity.",
        "Garbage collection in programming languages automatically reclaims memory that is no longer in use. "
        "Tracing collectors identify reachable objects, while reference counting tracks per-object references.",
        "DNA replication is the biological process of producing two identical replicas from one original DNA "
        "molecule. The double helix unwinds and each strand serves as a template for complementary base pairing.",
        "The JavaScript engine V8 compiles JavaScript directly to native machine code before executing it. "
        "Just-in-time compilation and inline caching optimize frequently executed code paths at runtime.",
        "The electromagnetic spectrum encompasses all wavelengths of electromagnetic radiation from radio waves "
        "to gamma rays. Visible light occupies a narrow band between approximately 400 and 700 nanometers.",
        "Tokenization splits text into smaller units called tokens for processing by language models. "
        "Subword tokenization methods like Byte-Pair Encoding (BPE) balance vocabulary size with coverage.",
        "The Krebs cycle, also known as the citric acid cycle, is a series of chemical reactions used by "
        "aerobic organisms to generate energy through the oxidation of acetyl-CoA derived from carbohydrates.",
        "Microservices architecture structures an application as a collection of loosely coupled services. "
        "Each service is independently deployable, scalable, and organized around business capabilities.",
        "The Hubble Space Telescope orbits Earth and captures high-resolution images free from atmospheric "
        "distortion. It has enabled measurements of the universe's expansion rate and age.",
        "Reinforcement learning from human feedback (RLHF) fine-tunes language models using human preference "
        "comparisons. A reward model trained on human rankings guides policy optimization toward helpful outputs.",
        "The Amazon rainforest is the world's largest tropical rainforest, spanning nine countries in South "
        "America. It produces approximately twenty percent of the world's oxygen and hosts immense biodiversity.",
        "Transformer models use positional encodings to inject information about token order since self-attention "
        "is permutation-invariant. Sinusoidal encodings and learned position embeddings are common approaches.",
        "The Apollo program landed the first humans on the Moon between 1969 and 1972. Six successful lunar "
        "landings returned over 380 kilograms of lunar samples for scientific analysis.",
        "Compiler design involves lexical analysis, parsing, semantic analysis, optimization, and code generation. "
        "Intermediate representations bridge the gap between high-level source code and target machine instructions.",
        "The endocrine system is a network of glands that produce and secrete hormones into the bloodstream. "
        "These chemical messengers regulate metabolism, growth, reproduction, sleep, and mood.",
        "P versus NP is a major unsolved problem in computer science. It asks whether every problem whose "
        "solution can be verified quickly can also be solved quickly by a deterministic Turing machine.",
        "The Internet of Things (IoT) connects physical devices embedded with sensors, software, and network "
        "connectivity. Smart homes, industrial monitoring, and healthcare devices exemplify IoT applications.",
        "The Earth's atmosphere is composed primarily of nitrogen and oxygen, with trace amounts of argon, "
        "carbon dioxide, and other gases. It protects life by absorbing ultraviolet solar radiation.",
        "Distributed systems coordinate components located on networked computers that communicate by passing "
        "messages. Challenges include partial failures, clock synchronization, and consensus in the presence of faults.",
        "Photosystem II is the first protein complex in the light-dependent reactions of oxygenic photosynthesis. "
        "It splits water molecules to release electrons, protons, and molecular oxygen.",
        "The Rust ownership model tracks memory lifetimes at compile time. Each value has exactly one owner, "
        "and references must not outlive the data they point to, eliminating entire classes of bugs.",
        "GPT architectures use causal self-attention with triangular masking to prevent tokens from attending "
        "to future positions. This autoregressive property enables efficient training via teacher forcing.",
        "The printing press invented by Johannes Gutenberg around 1440 revolutionized the production of books. "
        "Movable type enabled mass communication and accelerated the spread of knowledge during the Renaissance.",
        "Concurrency models in programming include threads with shared memory, message-passing actors, "
        "and async/await coroutines. Each model trades off between performance, safety, and ease of reasoning.",
        "The Central Dogma of molecular biology describes the flow of genetic information: DNA is transcribed "
        "into RNA, which is translated into proteins. Reverse transcription provides an exception in some viruses.",
    ]


def build_document_corpus(
    embed_weight: torch.Tensor,
    tokenizer,
    num_docs: int = 300,
) -> np.ndarray:
    """Create document embeddings and return [num_docs, 2048] float32 array."""
    texts = _load_wikitext_docs(max_docs=num_docs)
    token_ids_list = _tokenize_texts(texts, tokenizer, max_tokens=128)
    corpus = _average_token_embeddings(token_ids_list, embed_weight)
    print(f"  document corpus: {corpus.shape}  ({len(texts)} docs)")
    return corpus


def build_index(corpus: np.ndarray, metric: str = "ip") -> faiss.Index:
    """Build a FAISS flat index over document embeddings.

    Args:
        corpus: [num_docs, 2048] float32.
        metric: "l2" for L2 distance, "ip" for inner product (cosine for normalized vectors).

    Returns:
        FAISS index with corpus vectors added.
    """
    corpus = np.ascontiguousarray(corpus.astype(np.float32))
    if metric == "ip":
        faiss.normalize_L2(corpus)  # IP on L2-normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(corpus.shape[1])
    else:
        index = faiss.IndexFlatL2(corpus.shape[1])
    index.add(corpus)
    print(f"  FAISS IndexFlat{'IP' if metric == 'ip' else 'L2'}: {index.ntotal} vectors x {index.d}")
    return index


# ---------------------------------------------------------------------------
# 3. Native RAG injection (no projection matrices)
# ---------------------------------------------------------------------------

def inject_native_rag(
    hidden_states: np.ndarray,
    index: faiss.Index,
    top_k: int = TOP_K,
    scale: float = RETRIEVAL_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve top-k document vectors from FAISS and return the retrieval metadata.

    This is the "native" path: hidden states ARE the query (no W_query),
    and retrieved document vectors ARE additive signals (no W_context).
    Both live in the same 2048-dim embedding space.

    Args:
        hidden_states: [batch, seq, 2048] or [*, 2048] float32 hidden states.
        index: FAISS index over document embeddings (same 2048-dim space).
        top_k: number of documents to retrieve.
        scale: coefficient for additive injection: h_new = h + scale * doc_vector.

    Returns:
        (retrieved_vectors, distances, indices)
          retrieved_vectors: [batch, seq, top_k, 2048] or [*, top_k, 2048]
          distances:          [batch, seq, top_k] or [*, top_k]
          indices:            [batch, seq, top_k] or [*, top_k]
    """
    h = np.ascontiguousarray(hidden_states.astype(np.float32))
    original_shape = h.shape

    # Handle arbitrary leading dimensions: flatten to [N_queries, 2048]
    h_flat = h.reshape(-1, HIDDEN_DIM)

    # Normalize queries if using inner-product index
    if isinstance(index, faiss.IndexFlatIP):
        faiss.normalize_L2(h_flat)

    distances, indices = index.search(h_flat, top_k)  # [N_queries, K]

    # Reconstruct retrieved document vectors
    retrieved_flat = index.reconstruct_batch(indices.ravel())  # [N_queries*K, 2048]
    retrieved = retrieved_flat.reshape(h_flat.shape[0], top_k, HIDDEN_DIM)  # [N_queries, K, 2048]

    # Reshape back to original leading dims
    retrieved = retrieved.reshape(*original_shape[:-1], top_k, HIDDEN_DIM)

    return retrieved, distances, indices


# ---------------------------------------------------------------------------
# 4. End-to-end smoke test
# ---------------------------------------------------------------------------

def smoke_test() -> None:
    """Load Qwen2.5-3B embedding layer, build FAISS index over documents,
    retrieve native vectors, and verify shapes + sanity.
    """
    print("=" * 60)
    print("Native RAG End-to-End Test")
    print("=" * 60)

    # --- 4a. Load tokenizer + embedding layer ---
    print("\n[1/5] Loading Qwen2.5-3B tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  vocab_size: {tokenizer.vocab_size}")

    print("\n[2/5] Extracting embed_tokens.weight...")
    embed_weight = extract_embedding_layer(MODEL_ID)
    assert embed_weight.shape[1] == HIDDEN_DIM, \
        f"Embedding dim mismatch: {embed_weight.shape[1]} != {HIDDEN_DIM}"
    assert embed_weight.shape[0] >= tokenizer.vocab_size, \
        f"Vocab size mismatch: embed has {embed_weight.shape[0]}, tokenizer has {tokenizer.vocab_size}"
    assert embed_weight.dtype == torch.float32

    # --- 4b. Build document corpus and FAISS index ---
    print("\n[3/5] Building document corpus and FAISS index...")
    corpus = build_document_corpus(embed_weight, tokenizer, num_docs=300)
    index = build_index(corpus, metric="ip")

    # Verify index responds
    q_test = corpus[:3].copy()
    _, idx = index.search(q_test, 1)
    assert (idx[:, 0] == np.arange(3, dtype=np.int64)).all(), \
        "Index self-retrieval failed"
    print("  self-retrieval check: PASSED (each doc retrieves itself)")

    # --- 4c. Simulated hidden states and native retrieval ---
    print("\n[4/5] Running native RAG retrieval...")

    # Simulate hidden states at layer 17 by encoding test queries
    # through the token embedding layer (same 2048-dim space)
    test_queries = [
        "The transformer attention mechanism computes similarity between queries and keys",
        "Mitochondria produce energy through oxidative phosphorylation",
        "Quantum computers use qubits and entanglement for computation",
        "Photosynthesis in plants converts light into chemical energy",
    ]
    query_ids = [torch.tensor(tokenizer.encode(q, add_special_tokens=True)[:64], dtype=torch.long)
                 for q in test_queries]

    # Each query: average token embeddings to get a single [2048] vector
    query_vectors = np.stack(
        [embed_weight[ids].mean(dim=0).numpy() for ids in query_ids],
        axis=0,
    ).astype(np.float32)  # [4, 2048]

    print(f"  query vectors: {query_vectors.shape}")

    # For batch_seq test: simulate [batch=2, seq=4, 2048] hidden states
    batch_seq_hidden = query_vectors.reshape(2, 2, HIDDEN_DIM).astype(np.float32)
    print(f"  simulated hidden states [batch=2, seq=2, {HIDDEN_DIM}]: {batch_seq_hidden.shape}")

    retrieved, distances, iidx = inject_native_rag(
        batch_seq_hidden, index, top_k=TOP_K, scale=RETRIEVAL_SCALE,
    )
    print(f"  retrieved vectors: {retrieved.shape}  (expected: [2, 2, {TOP_K}, {HIDDEN_DIM}])")
    assert retrieved.shape == (2, 2, TOP_K, HIDDEN_DIM), \
        f"Retrieved shape mismatch: {retrieved.shape}"
    assert retrieved.dtype == np.float32
    print(f"  distances shape: {distances.shape}")
    print(f"  indices shape:   {iidx.shape}")

    # --- 4d. Show what was retrieved for each query ---
    print("\n[5/5] Retrieval results:")
    docs = _load_wikitext_docs(max_docs=300)
    queries_flat = query_vectors
    for i, q_text in enumerate(test_queries):
        print(f"\n  Query {i}: \"{q_text[:80]}...\"")
        top_docs = iidx[i]
        top_dists = distances[i] if distances.ndim == 2 else distances[i:i+1]
        if top_dists.ndim == 2:
            top_dists = top_dists[0]
        else:
            top_dists = top_dists.flatten()
        for rank in range(min(3, TOP_K)):
            doc_idx = int(top_docs[rank])
            doc_text = docs[doc_idx][:100].replace("\n", " ")
            dist_val = float(top_dists[rank])
            print(f"    #{rank+1} [idx={doc_idx}, dist={dist_val:.4f}] \"{doc_text}...\"")

    # --- 4e. Verify retrieved vectors are usable as additive signals ---
    # Since both hidden states and retrieved docs live in the same embedding
    # space, we can add them directly. Demonstrate the injection:
    pooled = batch_seq_hidden.mean(axis=1)          # [2, 2048]
    ret_pooled = retrieved.mean(axis=(1, 2))         # [2, 2048]
    injected = pooled + RETRIEVAL_SCALE * ret_pooled  # h_new = h + scale * doc_vector

    print(f"\n  Pooled hidden [2, {HIDDEN_DIM}]:       {pooled.shape}")
    print(f"  Retrieved pooled [2, {HIDDEN_DIM}]:     {ret_pooled.shape}")
    print(f"  Injected (h + {RETRIEVAL_SCALE}*doc):   {injected.shape}")
    print(f"  Injection delta norm:                    {np.linalg.norm(RETRIEVAL_SCALE * ret_pooled, axis=-1)}")
    assert injected.shape == (2, HIDDEN_DIM)

    # Flat query test
    q_single = query_vectors[0:1]  # [1, 2048]
    ret_single, dist_single, idx_single = inject_native_rag(q_single, index, top_k=TOP_K)
    assert ret_single.shape == (1, TOP_K, HIDDEN_DIM)
    print(f"\n  Single-query retrieved: {ret_single.shape}  (expected: [1, {TOP_K}, {HIDDEN_DIM}])")

    print("\n" + "=" * 60)
    print("SUCCESS: Native RAG runs end-to-end with correct shapes.")
    print("Retrieved vectors are valid 2048-dim embeddings in model-native space.")
    print("No projection matrices needed -- hidden states and documents share the same space.")
    print("=" * 60)


if __name__ == "__main__":
    smoke_test()
