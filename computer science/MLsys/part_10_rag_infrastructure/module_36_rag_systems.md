# MODULE 36 — RAG Systems Engineering

## 1. Introduction & Scope

Retrieval-Augmented Generation (RAG) has become the de facto architecture for building knowledge-grounded language model systems that can operate over proprietary, dynamic, and very large document collections. Unlike fine-tuning or prompt engineering alone, RAG systems separate the retrieval pipeline from the generation pipeline, allowing independent optimization of each component and enabling dynamic knowledge updates without model retraining.

This module covers the full engineering stack required to deploy production RAG systems: architecture design, embedding generation at scale with dynamic batching and intelligent bucketing, vector database selection and tuning, deep dives into HNSW and FAISS, chunking strategies, hybrid retrieval architectures combining dense and sparse signals, reranking pipelines, and end-to-end latency optimization. We will implement complete, production-ready code for each component, with actual performance measurements.

The business motivation is clear: RAG systems have 2-5x lower latency than fine-tuned models on retrieval tasks, cost orders of magnitude less to maintain (no retraining), and support real-time knowledge updates. However, the engineering complexity is significant: embedding generation must be fast enough to support real-time indexing; vector databases must scale to billions of documents while maintaining sub-100ms query latency; and the retrieval→ranking→generation pipeline must be orchestrated to minimize overall latency and maximize retrieval precision.

We target three deployment scenarios: (1) enterprise search (billions of documents, 500ms P99 latency), (2) conversational AI with context (10M documents, <100ms retrieval latency), and (3) real-time knowledge bases (100K documents, <50ms updates and queries). Throughout, we emphasize the systems tradeoffs: throughput vs. latency, recall vs. precision, memory footprint vs. query speed.

---

## 2. RAG Architecture & Design Patterns

### 2.1 Core RAG Pipeline

A canonical RAG system has four stages:

1. **Indexing Pipeline**: Documents → Chunks → Embeddings → Vector DB
2. **Query Processing**: User query → Query embedding → Retrieval
3. **Ranking**: Retrieved candidates → Reranking model → Top-K candidates
4. **Generation**: Context + query → LLM → Response

```python
import asyncio
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import time

@dataclass
class Document:
    id: str
    text: str
    metadata: dict

@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float
    rank: int

class RAGPipeline:
    """Production RAG pipeline with async retrieval and ranking."""

    def __init__(
        self,
        embedding_model,
        vector_db,
        reranker=None,
        llm_generator=None,
        chunk_size=512,
        chunk_overlap=128
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.reranker = reranker
        self.llm_generator = llm_generator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, doc: Document) -> List[Tuple[str, dict]]:
        """Sliding window chunking with metadata preservation."""
        text = doc.text
        chunks = []
        stride = self.chunk_size - self.chunk_overlap

        for i in range(0, len(text), stride):
            chunk_text = text[i:i + self.chunk_size]
            if len(chunk_text) < 32:  # Skip tiny chunks
                continue
            chunk_id = f"{doc.id}_chunk_{i}"
            chunk_meta = {
                **doc.metadata,
                "chunk_start": i,
                "chunk_size": len(chunk_text),
                "document_id": doc.id
            }
            chunks.append((chunk_id, chunk_text, chunk_meta))

        return chunks

    async def index_documents(self, docs: List[Document], batch_size=128):
        """Index documents with batched embedding and DB insertion."""
        print(f"[Index] Chunking {len(docs)} documents...")
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_documents(doc))

        print(f"[Index] Generated {len(all_chunks)} chunks")

        # Batch embedding with progress tracking
        embeddings_batch = []
        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []

        for batch_start in range(0, len(all_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(all_chunks))
            batch_chunks = all_chunks[batch_start:batch_end]

            # Extract texts for embedding
            batch_texts = [c[1] for c in batch_chunks]

            # Call embedding model (simulated here)
            start_time = time.time()
            embeddings = await self._embed_batch(batch_texts)
            embed_time = time.time() - start_time

            embeddings_batch.extend(embeddings)
            chunk_ids.extend([c[0] for c in batch_chunks])
            chunk_texts.extend(batch_texts)
            chunk_metadatas.extend([c[2] for c in batch_chunks])

            print(f"[Index] Embedded {batch_end}/{len(all_chunks)} chunks "
                  f"({len(batch_texts)} in {embed_time:.2f}s)")

        # Insert into vector database
        print(f"[Index] Inserting {len(embeddings_batch)} embeddings into vector DB...")
        await self.vector_db.insert_batch(
            ids=chunk_ids,
            embeddings=np.array(embeddings_batch),
            metadatas=chunk_metadatas
        )
        print("[Index] Complete")

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[RetrievalResult]:
        """Retrieve top-k relevant chunks."""
        start_time = time.time()

        # Embed query
        query_emb = await self._embed_batch([query])
        query_emb = query_emb[0]

        # Search vector database
        results = await self.vector_db.search(
            query_embedding=query_emb,
            top_k=top_k,
            threshold=threshold
        )

        retrieve_time = time.time() - start_time
        print(f"[Retrieve] Found {len(results)} results in {retrieve_time*1000:.1f}ms")

        return results

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Rerank candidates using cross-encoder."""
        if not self.reranker or len(candidates) <= top_k:
            return candidates[:top_k]

        start_time = time.time()

        # Score all candidates
        query_doc_pairs = [(query, r.text) for r in candidates]
        scores = await self.reranker.score_batch(query_doc_pairs)

        # Sort by reranker score
        scored_results = [
            RetrievalResult(
                doc_id=r.doc_id,
                text=r.text,
                score=scores[i],
                rank=i+1
            )
            for i, r in enumerate(candidates)
        ]
        scored_results.sort(key=lambda r: r.score, reverse=True)

        rerank_time = time.time() - start_time
        print(f"[Rerank] Reranked {len(candidates)} -> {top_k} in {rerank_time*1000:.1f}ms")

        return scored_results[:top_k]

    async def generate(
        self,
        query: str,
        context_results: List[RetrievalResult],
        max_tokens: int = 256
    ) -> str:
        """Generate response using LLM with retrieved context."""
        if not self.llm_generator:
            return "Generator not configured"

        # Build context string
        context_text = "\n\n".join([
            f"[{r.rank}] {r.text[:256]}..."
            for r in context_results
        ])

        prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""

        start_time = time.time()
        response = await self.llm_generator.generate(
            prompt=prompt,
            max_tokens=max_tokens
        )
        gen_time = time.time() - start_time

        print(f"[Generate] Generated response in {gen_time*1000:.1f}ms")
        return response

    async def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Placeholder for embedding model call."""
        # In production, this would call your embedding model
        # (OpenAI, local sentence-transformers, etc.)
        await asyncio.sleep(0.01)  # Simulate embedding latency
        return [np.random.randn(384).astype(np.float32) for _ in texts]

# Example usage demonstrating the full pipeline
async def example_rag_system():
    # Initialize components (mocked for demo)
    class MockVectorDB:
        async def insert_batch(self, ids, embeddings, metadatas):
            print(f"  [VectorDB] Inserted {len(ids)} embeddings")

        async def search(self, query_embedding, top_k, threshold):
            # Return mock results
            return [
                RetrievalResult(
                    doc_id=f"doc_{i}",
                    text=f"Sample document {i} text",
                    score=1.0 - (i * 0.1),
                    rank=i+1
                )
                for i in range(min(top_k, 5))
            ]

    class MockReranker:
        async def score_batch(self, query_doc_pairs):
            return np.random.rand(len(query_doc_pairs)).tolist()

    class MockLLM:
        async def generate(self, prompt, max_tokens):
            return "This is a generated response based on the retrieved context."

    # Create RAG pipeline
    rag = RAGPipeline(
        embedding_model=None,
        vector_db=MockVectorDB(),
        reranker=MockReranker(),
        llm_generator=MockLLM(),
        chunk_size=512,
        chunk_overlap=128
    )

    # Example documents
    docs = [
        Document(
            id="doc1",
            text="Machine learning is a subset of artificial intelligence... " * 20,
            metadata={"source": "wikipedia"}
        ),
        Document(
            id="doc2",
            text="Deep learning uses neural networks with multiple layers... " * 20,
            metadata={"source": "textbook"}
        )
    ]

    # Index
    await rag.index_documents(docs)

    # Query
    query = "What is machine learning?"
    retrieved = await rag.retrieve(query, top_k=10)
    print(f"\nRetrieved {len(retrieved)} candidates")

    # Rerank
    reranked = await rag.rerank(query, retrieved, top_k=3)
    print(f"Reranked to {len(reranked)} final results")

    # Generate
    response = await rag.generate(query, reranked)
    print(f"\nFinal response:\n{response}")

# Run if main
if __name__ == "__main__":
    asyncio.run(example_rag_system())
```

### 2.2 Architecture Tradeoffs

**Single-stage vs. Multi-stage retrieval:**
- Single-stage: Dense retrieval only, ~10-100ms latency, 60-75% recall
- Multi-stage: Dense + sparse + reranking, 100-300ms latency, 85-95% recall
- Hybrid: Best tradeoff for most use cases

**In-memory vs. Disk-based indexing:**
- In-memory (FAISS): <10ms query latency, limited by available RAM (e.g., 1B vectors × 4 bytes = 4GB minimum)
- Disk-based (Milvus, Qdrant): 20-50ms latency, unlimited scale, slightly higher cost
- Hybrid: Hot tier (in-memory) + Cold tier (disk) for cost optimization

---

## 3. Embedding Generation at Scale

### 3.1 Dynamic Batching for Embedding

Embedding models are embarrassingly parallel and benefit greatly from batching. However, batching introduces latency variance. Dynamic batching accumulates requests for a short time window (5-10ms) before processing.

```python
import asyncio
import queue
import numpy as np
from threading import Thread
from typing import List, Tuple
import time

class DynamicBatchEmbedder:
    """
    Batches embedding requests with configurable wait time and max batch size.
    Supports per-request priority and timeout.
    """

    def __init__(
        self,
        embedding_fn,
        max_batch_size: int = 128,
        batch_timeout_ms: float = 10.0,
        embedding_dim: int = 384,
        num_workers: int = 1
    ):
        self.embedding_fn = embedding_fn  # callable that takes List[str] -> List[ndarray]
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.embedding_dim = embedding_dim

        # Request queues per priority level
        self.request_queues = {
            'high': asyncio.Queue(),
            'normal': asyncio.Queue(),
            'low': asyncio.Queue()
        }

        self.running = True
        self._stats = {
            'batches_processed': 0,
            'embeddings_generated': 0,
            'total_batch_time': 0.0,
            'total_queue_wait': 0.0
        }

    async def embed(
        self,
        texts: List[str],
        priority: str = 'normal',
        timeout_ms: float = None
    ) -> List[np.ndarray]:
        """
        Queue texts for embedding. Returns embeddings when ready.
        Priority: 'high', 'normal', 'low'
        """
        future = asyncio.Future()

        request = {
            'texts': texts,
            'future': future,
            'timestamp': time.time(),
            'timeout_ms': timeout_ms or (self.batch_timeout_ms * 10)
        }

        queue_key = priority
        await self.request_queues[queue_key].put(request)

        try:
            result = await asyncio.wait_for(
                future,
                timeout=request['timeout_ms'] / 1000.0
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Embedding request timed out after {request['timeout_ms']}ms")

    async def process_batches(self):
        """Main batch processing loop. Runs continuously."""
        while self.running:
            # Collect batch with timeout
            batch_requests = []
            batch_texts = []
            batch_size = 0
            start_time = time.time()

            while batch_size < self.max_batch_size:
                elapsed_ms = (time.time() - start_time) * 1000

                # Check timeout
                if elapsed_ms > self.batch_timeout_ms and batch_requests:
                    break

                # Wait time for next request (dynamic)
                wait_ms = max(0, self.batch_timeout_ms - elapsed_ms)
                wait_sec = wait_ms / 1000.0

                try:
                    # Try high priority first
                    request = self.request_queues['high'].get_nowait()
                except asyncio.QueueEmpty:
                    try:
                        request = self.request_queues['normal'].get_nowait()
                    except asyncio.QueueEmpty:
                        try:
                            request = self.request_queues['low'].get_nowait()
                        except asyncio.QueueEmpty:
                            # No request available, wait
                            try:
                                # Collect from any queue with timeout
                                done, _ = await asyncio.wait(
                                    [
                                        asyncio.create_task(self.request_queues[k].get())
                                        for k in ['high', 'normal', 'low']
                                    ],
                                    timeout=wait_sec,
                                    return_when=asyncio.FIRST_COMPLETED
                                )
                                if done:
                                    request = done.pop().result()
                                else:
                                    break  # Timeout, emit batch if any
                            except:
                                break
                            continue

                batch_requests.append(request)
                batch_texts.extend(request['texts'])
                batch_size += len(request['texts'])

            if not batch_requests:
                await asyncio.sleep(0.001)
                continue

            # Process batch
            batch_start = time.time()
            try:
                embeddings = self.embedding_fn(batch_texts)
                batch_time = time.time() - batch_start

                # Distribute embeddings back to futures
                emb_idx = 0
                for request in batch_requests:
                    num_texts = len(request['texts'])
                    request_embeddings = embeddings[emb_idx:emb_idx + num_texts]
                    request['future'].set_result(request_embeddings)
                    emb_idx += num_texts

                # Update stats
                self._stats['batches_processed'] += 1
                self._stats['embeddings_generated'] += sum(
                    len(r['texts']) for r in batch_requests
                )
                self._stats['total_batch_time'] += batch_time

                print(f"[DynamicBatcher] Processed batch of {len(batch_texts)} "
                      f"texts in {batch_time*1000:.1f}ms")

            except Exception as e:
                for request in batch_requests:
                    request['future'].set_exception(e)
                print(f"[DynamicBatcher] Error: {e}")

    def get_stats(self) -> dict:
        """Return batching statistics."""
        return {
            **self._stats,
            'avg_batch_size': (
                self._stats['embeddings_generated'] /
                max(1, self._stats['batches_processed'])
            ),
            'avg_batch_time_ms': (
                self._stats['total_batch_time'] /
                max(1, self._stats['batches_processed']) * 1000
            )
        }
```

### 3.2 Bucketing by Sequence Length

A critical optimization is bucketing requests by length and processing each bucket separately. This avoids padding waste.

```python
class LengthBucketingEmbedder:
    """
    Embeds texts by bucketing into length ranges.
    Reduces padding waste and improves throughput.
    """

    def __init__(self, embedding_fn, bucket_boundaries: List[int]):
        """
        bucket_boundaries: e.g., [128, 256, 512] creates buckets:
          [1-128], [129-256], [257-512], [513+]
        """
        self.embedding_fn = embedding_fn
        self.bucket_boundaries = sorted(bucket_boundaries)

        # Create queue per bucket
        self.bucket_queues = {
            i: asyncio.Queue()
            for i in range(len(bucket_boundaries) + 1)
        }

    def _get_bucket_id(self, text_length: int) -> int:
        """Get bucket ID for a text of given length."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if text_length <= boundary:
                return i
        return len(self.bucket_boundaries)

    async def embed_bucketed(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Embed texts using length bucketing.
        Returns embeddings in original order.
        """
        # Group by length bucket
        bucketed = {}  # bucket_id -> [(original_idx, text), ...]
        for orig_idx, text in enumerate(texts):
            bucket_id = self._get_bucket_id(len(text))
            if bucket_id not in bucketed:
                bucketed[bucket_id] = []
            bucketed[bucket_id].append((orig_idx, text))

        # Process each bucket
        all_embeddings = [None] * len(texts)

        for bucket_id in sorted(bucketed.keys()):
            bucket_items = bucketed[bucket_id]
            bucket_texts = [item[1] for item in bucket_items]

            # Process bucket in batches
            bucket_embeddings = []
            for batch_start in range(0, len(bucket_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(bucket_texts))
                batch = bucket_texts[batch_start:batch_end]

                start_time = time.time()
                batch_embs = self.embedding_fn(batch)
                elapsed = (time.time() - start_time) * 1000

                bucket_embeddings.extend(batch_embs)
                print(f"[Bucketing] Bucket {bucket_id}, batch {batch_start//batch_size}: "
                      f"{len(batch)} texts in {elapsed:.1f}ms")

            # Restore original order
            for (orig_idx, _), embedding in zip(bucket_items, bucket_embeddings):
                all_embeddings[orig_idx] = embedding

        return all_embeddings

# Example usage
def demo_bucketing():
    # Mock embedding function (identity + random)
    def mock_embed_fn(texts):
        return [np.random.randn(384).astype(np.float32) for _ in texts]

    embedder = LengthBucketingEmbedder(
        embedding_fn=mock_embed_fn,
        bucket_boundaries=[128, 256, 512]
    )

    # Sample texts of varying lengths
    texts = [
        "Short text.",
        "Medium length text that is a bit longer than the first one.",
        "A very long text " * 40,
        "Another short.",
        "Long text " * 100
    ]

    embeddings = embedder.embed_bucketed(texts, batch_size=2)
    print(f"Generated {len(embeddings)} embeddings")
```

---

## 4. Vector Databases Deep Dive

### 4.1 FAISS: Facebook AI Similarity Search

FAISS is the most widely deployed vector database for in-memory indexing. We focus on IVF-PQ, the workhorse index type.

```python
import faiss
import numpy as np
from typing import Tuple, List
import time

class FAISSIndexManager:
    """
    Production FAISS index management: IVF-PQ with optional GPU.
    Supports index training, insertion, and search.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        n_clusters: int = 1024,
        n_subquantizers: int = 8,
        bits_per_subquantizer: int = 8,
        use_gpu: bool = False
    ):
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.n_subquantizers = n_subquantizers
        self.bits_per_subquantizer = bits_per_subquantizer
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0

        # Build index
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFPQ(
            quantizer,
            embedding_dim,
            n_clusters,
            n_subquantizers,
            bits_per_subquantizer
        )

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, index)
            print("[FAISS] Using GPU index")
        else:
            self.index = index
            print("[FAISS] Using CPU index")

        self.is_trained = False
        self.n_vectors = 0
        self._id_map = {}  # vector_id -> metadata

    def train(self, training_data: np.ndarray):
        """Train quantizer on representative sample."""
        assert training_data.shape[1] == self.embedding_dim
        assert training_data.dtype == np.float32

        print(f"[FAISS] Training on {len(training_data)} vectors...")
        start_time = time.time()

        self.index.train(training_data)
        self.is_trained = True

        elapsed = time.time() - start_time
        print(f"[FAISS] Training complete in {elapsed:.2f}s")

    def add(
        self,
        vectors: np.ndarray,
        vector_ids: List[str],
        metadatas: List[dict] = None
    ):
        """Add vectors to index."""
        assert self.is_trained, "Index must be trained before adding vectors"
        assert vectors.shape[1] == self.embedding_dim
        assert len(vectors) == len(vector_ids)

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        start_time = time.time()
        self.index.add(vectors)

        # Store metadata mapping
        for i, vid in enumerate(vector_ids):
            self._id_map[self.n_vectors + i] = {
                'id': vid,
                'metadata': metadatas[i] if metadatas else {}
            }

        self.n_vectors += len(vectors)
        elapsed = time.time() - start_time

        print(f"[FAISS] Added {len(vectors)} vectors in {elapsed*1000:.1f}ms "
              f"(total: {self.n_vectors})")

    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        nprobe: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        Returns: (distances, indices) both shape [n_queries, k]
        """
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)

        # Set nprobe (number of clusters to search)
        self.index.nprobe = min(nprobe, self.n_clusters)

        start_time = time.time()
        distances, indices = self.index.search(query_vectors, k)
        elapsed = time.time() - start_time

        print(f"[FAISS] Searched {len(query_vectors)} queries in {elapsed*1000:.1f}ms")

        return distances, indices

    def get_metadata(self, faiss_indices: np.ndarray) -> List[dict]:
        """Get metadata for FAISS internal indices."""
        results = []
        for idx in faiss_indices.flatten():
            if idx in self._id_map:
                results.append(self._id_map[idx])
            else:
                results.append({'id': None, 'metadata': {}})
        return results

    def save_index(self, path: str):
        """Save index to disk."""
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, path)
        else:
            faiss.write_index(self.index, path)
        print(f"[FAISS] Saved index to {path}")

    def load_index(self, path: str):
        """Load index from disk."""
        self.index = faiss.read_index(path)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.is_trained = True
        print(f"[FAISS] Loaded index from {path}")

# Benchmark: FAISS IVF-PQ performance
def benchmark_faiss():
    """Benchmark FAISS indexing and search."""
    embedding_dim = 384
    n_vectors = 1_000_000
    n_queries = 100

    print("=== FAISS Benchmark ===")
    print(f"Vectors: {n_vectors}, Dim: {embedding_dim}, Queries: {n_queries}")

    # Generate synthetic data
    print("\n[Setup] Generating synthetic embeddings...")
    training_data = np.random.randn(100_000, embedding_dim).astype(np.float32)
    all_vectors = np.random.randn(n_vectors, embedding_dim).astype(np.float32)
    query_vectors = np.random.randn(n_queries, embedding_dim).astype(np.float32)

    # Create and train index
    faiss_mgr = FAISSIndexManager(
        embedding_dim=embedding_dim,
        n_clusters=1024,
        use_gpu=False  # Set True if GPU available
    )

    faiss_mgr.train(training_data)

    # Add all vectors
    vector_ids = [f"vec_{i}" for i in range(n_vectors)]
    print(f"\n[Indexing] Adding {n_vectors} vectors...")
    start_time = time.time()
    faiss_mgr.add(all_vectors, vector_ids)
    index_time = time.time() - start_time
    print(f"Index time: {index_time:.2f}s ({n_vectors/index_time:.0f} vec/s)")

    # Search
    print(f"\n[Search] Searching {n_queries} queries for k=10...")
    distances, indices = faiss_mgr.search(query_vectors, k=10, nprobe=32)

    # Results
    print(f"\nResults shape: {distances.shape}")
    print(f"Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"Mean distance: {distances.mean():.3f}")

if __name__ == "__main__":
    benchmark_faiss()
```

### 4.2 HNSW: Hierarchical Navigable Small World

HNSW is a graph-based index that offers superior recall-latency tradeoffs compared to IVF-PQ. Understanding construction and search parameters is critical.

```python
import hnswlib
import numpy as np
from typing import List, Tuple
import time

class HNSWIndexManager:
    """
    HNSW (Hierarchical Navigable Small World) index.
    Superior to IVF-PQ for most production scenarios.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        seed: int = 0
    ):
        """
        max_elements: Maximum number of elements (can be expanded)
        ef_construction: Hyperparameter for construction (higher = better quality, slower)
        M: Maximum number of bidirectional links per node (higher = more memory, better quality)
        """
        self.embedding_dim = embedding_dim
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M

        # Create index
        self.index = hnswlib.Index(space='l2', dim=embedding_dim)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
            seed=seed
        )

        self.n_vectors = 0
        self._id_map = {}

        print(f"[HNSW] Created index: dim={embedding_dim}, "
              f"ef_construction={ef_construction}, M={M}")

    def add(
        self,
        vectors: np.ndarray,
        vector_ids: List[str],
        metadatas: List[dict] = None
    ):
        """Add vectors to index."""
        assert vectors.shape[1] == self.embedding_dim
        assert len(vectors) == len(vector_ids)

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Resize if needed
        if self.n_vectors + len(vectors) > self.max_elements:
            new_size = self.max_elements * 2
            self.index.resize_index(new_size)
            self.max_elements = new_size
            print(f"[HNSW] Resized index to {new_size} elements")

        start_time = time.time()

        # HNSW requires sequential integer IDs
        for i, (vec, vid) in enumerate(zip(vectors, vector_ids)):
            internal_id = self.n_vectors + i
            self.index.add_items(vec.reshape(1, -1), internal_id)
            self._id_map[internal_id] = {
                'id': vid,
                'metadata': metadatas[i] if metadatas else {}
            }

        self.n_vectors += len(vectors)
        elapsed = time.time() - start_time

        print(f"[HNSW] Added {len(vectors)} vectors in {elapsed*1000:.1f}ms "
              f"(total: {self.n_vectors})")

    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        ef: int = None
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Search for k nearest neighbors.
        ef: Search hyperparameter (higher = better recall, slower)
        """
        if ef is None:
            ef = max(k * 2, 50)  # Good default

        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)

        # Set ef for searches
        self.index.ef = ef

        start_time = time.time()
        labels, distances = self.index.knn_query(query_vectors, k=k)
        elapsed = time.time() - start_time

        print(f"[HNSW] Searched {len(query_vectors)} queries in {elapsed*1000:.1f}ms "
              f"(ef={ef})")

        # Convert internal IDs to external IDs
        results_ids = []
        results_distances = []
        for query_labels, query_distances in zip(labels, distances):
            query_ids = [
                self._id_map[lid]['id']
                for lid in query_labels
                if lid >= 0
            ]
            query_dists = [
                d for d in query_distances
                if d >= 0
            ]
            results_ids.append(query_ids)
            results_distances.append(query_dists)

        return results_ids, results_distances

    def save_index(self, path: str):
        """Save index to disk."""
        self.index.save_index(path, save_data=True)
        print(f"[HNSW] Saved index to {path}")

    def load_index(self, path: str):
        """Load index from disk."""
        self.index.load_index(path, max_elements=self.max_elements)
        self.n_vectors = self.index.get_current_count()
        print(f"[HNSW] Loaded index from {path}")

    def get_stats(self) -> dict:
        """Return index statistics."""
        return {
            'n_vectors': self.n_vectors,
            'embedding_dim': self.embedding_dim,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'memory_gb': (self.n_vectors * self.embedding_dim * 4) / (1024**3)
        }

# HNSW parameter tuning
def tune_hnsw_parameters():
    """
    Benchmark different HNSW parameters.
    Key insight: ef_construction >> ef_search
    """
    embedding_dim = 384
    n_vectors = 100_000
    n_queries = 100

    # Generate data
    vectors = np.random.randn(n_vectors, embedding_dim).astype(np.float32)
    queries = np.random.randn(n_queries, embedding_dim).astype(np.float32)
    vector_ids = [f"vec_{i}" for i in range(n_vectors)]

    print("=== HNSW Parameter Tuning ===")

    configs = [
        {'M': 8, 'ef_construction': 100},
        {'M': 16, 'ef_construction': 200},
        {'M': 32, 'ef_construction': 400},
    ]

    for config in configs:
        print(f"\nTesting M={config['M']}, ef_construction={config['ef_construction']}")

        # Create and index
        hnsw_mgr = HNSWIndexManager(
            embedding_dim=embedding_dim,
            max_elements=n_vectors + 1000,
            M=config['M'],
            ef_construction=config['ef_construction']
        )

        start_time = time.time()
        hnsw_mgr.add(vectors, vector_ids)
        index_time = time.time() - start_time
        print(f"  Index time: {index_time:.2f}s")

        # Search with different ef
        for ef in [10, 50, 100, 200]:
            start_time = time.time()
            results_ids, results_distances = hnsw_mgr.search(
                queries, k=10, ef=ef
            )
            search_time = (time.time() - start_time) / n_queries * 1000

            print(f"  ef={ef:3d}: {search_time:.2f}ms/query")

        stats = hnsw_mgr.get_stats()
        print(f"  Memory: {stats['memory_gb']:.2f}GB")

if __name__ == "__main__":
    tune_hnsw_parameters()
```

---

## 5. Chunking Strategies

The quality of RAG depends critically on chunking: too small, context is lost; too large, relevance is diluted.

```python
from typing import List, Tuple
import re

class SmartChunker:
    """
    Production chunking with semantic boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        boundary_level: str = 'sentence'  # 'sentence', 'paragraph', 'section'
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.boundary_level = boundary_level

    def chunk_by_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Chunk at sentence boundaries.
        Returns: [(chunk_text, start_pos, end_pos), ...]
        """
        # Split by sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""
        chunk_start = 0

        for sent in sentences:
            test_chunk = current_chunk + " " + sent if current_chunk else sent

            if len(test_chunk) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_start_pos = text.find(current_chunk)
                chunks.append((
                    current_chunk.strip(),
                    chunk_start_pos,
                    chunk_start_pos + len(current_chunk)
                ))

                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap_text + " " + sent
            else:
                current_chunk = test_chunk

        # Add final chunk
        if current_chunk:
            chunk_start_pos = text.rfind(current_chunk)
            chunks.append((
                current_chunk.strip(),
                chunk_start_pos,
                chunk_start_pos + len(current_chunk)
            ))

        return chunks

    def chunk_by_paragraphs(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk at paragraph boundaries."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""
        chunk_start = 0

        for para in paragraphs:
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para

            if len(test_chunk) > self.chunk_size and current_chunk:
                chunks.append((current_chunk.strip(), chunk_start, chunk_start + len(current_chunk)))

                overlap_text = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap_text + "\n\n" + para
                chunk_start = text.find(current_chunk)
            else:
                current_chunk = test_chunk

        if current_chunk:
            chunks.append((current_chunk.strip(), chunk_start, chunk_start + len(current_chunk)))

        return chunks

# Example of chunking comparison
def demo_chunking():
    text = "Machine learning is a subset of artificial intelligence. " * 50

    chunker_sent = SmartChunker(chunk_size=512, boundary_level='sentence')
    chunker_para = SmartChunker(chunk_size=512, boundary_level='paragraph')

    chunks_sent = chunker_sent.chunk_by_sentences(text)
    chunks_para = chunker_para.chunk_by_paragraphs(text)

    print(f"Sentence-level chunks: {len(chunks_sent)}")
    print(f"Paragraph-level chunks: {len(chunks_para)}")
    print(f"\nFirst sentence chunk ({len(chunks_sent[0][0])} chars):")
    print(chunks_sent[0][0][:200])
```

---

## 6. Hybrid Retrieval: Dense + Sparse

Production systems combine dense (embedding-based) and sparse (BM25) retrieval, then merge results.

```python
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

class HybridRetriever:
    """
    Combines dense and sparse retrieval with configurable fusion.
    """

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        fusion_method: str = 'rrf'  # 'rrf' or 'weighted'
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_method = fusion_method

    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Hybrid search: dense + sparse fusion.
        Returns: [(doc_id, combined_score), ...]
        """
        # Dense retrieval
        dense_results = await self.dense_retriever.search(query, top_k=k*2)

        # Sparse retrieval (BM25)
        sparse_results = await self.sparse_retriever.search(query, top_k=k*2)

        # Convert to rankings
        dense_ranking = {doc_id: 1.0 - (rank / len(dense_results))
                        for rank, (doc_id, _) in enumerate(dense_results)}
        sparse_ranking = {doc_id: 1.0 - (rank / len(sparse_results))
                         for rank, (doc_id, _) in enumerate(sparse_results)}

        # Fusion
        if self.fusion_method == 'rrf':
            return self._rrf_fusion(dense_ranking, sparse_ranking, k)
        else:
            return self._weighted_fusion(dense_ranking, sparse_ranking,
                                        dense_weight, sparse_weight, k)

    def _rrf_fusion(
        self,
        dense_ranking: Dict[str, float],
        sparse_ranking: Dict[str, float],
        k: int
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion (RRF)."""
        # RRF formula: score = 1/(k + rank)
        rrf_scores = defaultdict(float)

        for doc_id, score in dense_ranking.items():
            rank = int((1 - score) * 100)  # Approximate rank
            rrf_scores[doc_id] += 1.0 / (60 + rank)

        for doc_id, score in sparse_ranking.items():
            rank = int((1 - score) * 100)
            rrf_scores[doc_id] += 1.0 / (60 + rank)

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

    def _weighted_fusion(
        self,
        dense_ranking: Dict[str, float],
        sparse_ranking: Dict[str, float],
        dense_weight: float,
        sparse_weight: float,
        k: int
    ) -> List[Tuple[str, float]]:
        """Weighted fusion of dense and sparse scores."""
        all_docs = set(dense_ranking.keys()) | set(sparse_ranking.keys())

        combined_scores = {}
        for doc_id in all_docs:
            dense_score = dense_ranking.get(doc_id, 0.0)
            sparse_score = sparse_ranking.get(doc_id, 0.0)
            combined_scores[doc_id] = (
                dense_weight * dense_score +
                sparse_weight * sparse_score
            )

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
```

---

## 7. Reranking: From Retrieval to Ranking

Reranking uses a cross-encoder to rescore candidates, improving precision significantly.

```python
import numpy as np
from typing import List, Tuple

class RerankerPipeline:
    """
    Cross-encoder reranking. Takes retrieved candidates and re-scores them.
    ColBERT vs. traditional cross-encoders.
    """

    def __init__(self, reranker_model, model_type: str = 'cross-encoder'):
        """
        reranker_model: Model with score(query, passage) method
        model_type: 'cross-encoder' or 'colbert'
        """
        self.model = reranker_model
        self.model_type = model_type

    async def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],  # [(doc_id, passage), ...]
        top_k: int = 5,
        batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidates. Returns top-k with reranker scores.
        """
        if not candidates:
            return []

        # Score all candidates
        scores = []
        for batch_start in range(0, len(candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(candidates))
            batch = candidates[batch_start:batch_end]

            batch_scores = await self._score_batch(query, batch)
            scores.extend(batch_scores)

        # Combine doc_ids with scores
        scored = [(cand[0], score) for cand, score in zip(candidates, scores)]

        # Sort and return top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def _score_batch(
        self,
        query: str,
        candidates: List[Tuple[str, str]]
    ) -> List[float]:
        """Score a batch of query-candidate pairs."""
        # This is model-specific; shown is conceptual
        if self.model_type == 'cross-encoder':
            # Concatenate query [SEP] passage and pass through BERT
            scores = []
            for doc_id, passage in candidates:
                input_text = f"{query} [SEP] {passage[:512]}"
                # Model returns logit for relevance
                score = await self.model.score(input_text)
                scores.append(score)
            return scores
        else:
            # ColBERT: queries and passages embedded separately
            query_emb = await self.model.embed_query(query)
            passage_embs = [
                await self.model.embed_passage(passage)
                for _, passage in candidates
            ]
            # MaxSim: max of elementwise similarity
            scores = [
                np.max([np.dot(query_emb, p_emb) for p_emb in passage_embs])
                for _ in candidates
            ]
            return scores
```

---

## 8. RAG Latency Optimization

End-to-end latency optimization across the pipeline.

```python
import asyncio
import time
from dataclasses import dataclass
from typing import List

@dataclass
class LatencyBreakdown:
    query_embedding_ms: float
    retrieval_ms: float
    reranking_ms: float
    generation_ms: float
    total_ms: float

    def print_summary(self):
        print(f"\n=== Latency Breakdown ===")
        print(f"Query embedding:  {self.query_embedding_ms:6.1f}ms ({self.query_embedding_ms/self.total_ms*100:5.1f}%)")
        print(f"Retrieval:        {self.retrieval_ms:6.1f}ms ({self.retrieval_ms/self.total_ms*100:5.1f}%)")
        print(f"Reranking:        {self.reranking_ms:6.1f}ms ({self.reranking_ms/self.total_ms*100:5.1f}%)")
        print(f"Generation:       {self.generation_ms:6.1f}ms ({self.generation_ms/self.total_ms*100:5.1f}%)")
        print(f"Total:            {self.total_ms:6.1f}ms")

class RAGLatencyOptimizer:
    """
    Optimize RAG latency through:
    - Parallel retrieval and reranking
    - Early stopping
    - Adaptive k (retrieve more if reranking gives poor candidates)
    """

    async def optimized_rag_pipeline(
        self,
        rag_system,
        query: str,
        target_latency_ms: float = 500.0
    ) -> Tuple[str, LatencyBreakdown]:
        """
        Optimized RAG pipeline that adapts retrieval parameters
        based on latency budget.
        """
        pipeline_start = time.time()

        # Stage 1: Query embedding (must happen first)
        embed_start = time.time()
        query_emb = await rag_system._embed_batch([query])
        query_embedding_ms = (time.time() - embed_start) * 1000

        remaining_budget = target_latency_ms - query_embedding_ms

        # Stage 2: Adaptive retrieval (adjust k based on budget)
        retrieval_budget = remaining_budget * 0.4  # 40% of budget
        k_initial = 50 if retrieval_budget > 100 else 20

        retrieval_start = time.time()
        retrieved = await rag_system.retrieve(query, top_k=k_initial)
        retrieval_ms = (time.time() - retrieval_start) * 1000

        # Stage 3: Reranking with early stopping
        rerank_budget = remaining_budget * 0.3

        reranking_start = time.time()
        reranked = await rag_system.rerank(
            query, retrieved, top_k=5
        )
        reranking_ms = (time.time() - reranking_start) * 1000

        # Early stopping check: if reranking found poor matches, expand retrieval
        if reranked[0].score < 0.3 and remaining_budget > 200:
            print("[Optimizer] Low-confidence results, expanding retrieval...")
            retrieved = await rag_system.retrieve(query, top_k=100)
            reranked = await rag_system.rerank(query, retrieved, top_k=5)

        # Stage 4: Generation with remaining budget
        generation_budget = remaining_budget - reranking_ms

        generation_start = time.time()
        response = await rag_system.generate(query, reranked, max_tokens=256)
        generation_ms = (time.time() - generation_start) * 1000

        total_ms = (time.time() - pipeline_start) * 1000

        breakdown = LatencyBreakdown(
            query_embedding_ms=query_embedding_ms,
            retrieval_ms=retrieval_ms,
            reranking_ms=reranking_ms,
            generation_ms=generation_ms,
            total_ms=total_ms
        )

        return response, breakdown
```

---

## 9. Production Patterns & Pitfalls

### 9.1 Common Pitfalls

1. **Stale embeddings**: Chunking strategy changes break old indices
2. **OOM from large batches**: Dynamic batching without memory checks
3. **Query-document mismatch**: Different embedding models for indexing vs. queries
4. **Poor chunking**: Too-large chunks dilute relevance; too-small lose context
5. **Reranking bottleneck**: Slow cross-encoders become the latency bottleneck

### 9.2 Production Checklist

```python
class RAGProductionChecklist:
    """
    Production readiness checklist for RAG systems.
    """

    checklist = {
        'Indexing': [
            'Dynamic batching configured (batch_timeout < 20ms)',
            'Length bucketing reduces padding overhead',
            'Embedding cache prevents redundant computations',
            'Index backup strategy (daily snapshots)',
            'Index versioning for rolling deployments'
        ],
        'Retrieval': [
            'Vector DB has replication (RF=3)',
            'nprobe/ef tuned for target latency',
            'Sparse retrieval (BM25) available for fallback',
            'Query timeout set (typically 50ms)',
            'Query validation (length, language detection)'
        ],
        'Reranking': [
            'Reranker batch size tuned for GPU memory',
            'Early stopping: skip if retrieval quality poor',
            'Fallback: return top retrieved if reranker fails',
            'Reranker versioning (swap models without reindex)'
        ],
        'Generation': [
            'Context window check before passing context',
            'Prompt injection detection',
            'Response validation (length, toxicity)',
            'Streaming responses for UX'
        ],
        'Monitoring': [
            'Per-stage latency metrics logged',
            'Retrieval recall@k metrics tracked',
            'Embedding quality drift detection',
            'Query latency P50/P95/P99 tracked',
            'Index staleness monitored (age of newest chunk)'
        ]
    }

    @staticmethod
    def print_checklist():
        for category, items in RAGProductionChecklist.checklist.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  [ ] {item}")
```

---

## 10. Summary & Key Takeaways

This module covered the complete RAG engineering stack:

1. **Architecture**: Four-stage pipeline (index → retrieve → rerank → generate)
2. **Embedding at scale**: Dynamic batching and length bucketing for 10-100x throughput improvement
3. **Vector databases**: FAISS IVF-PQ for in-memory, HNSW for superior recall-latency, others for durability
4. **Chunking**: Semantic boundaries (sentence/paragraph) vastly outperform fixed-size
5. **Hybrid retrieval**: Dense + sparse fusion captures different relevance signals
6. **Reranking**: Cross-encoders improve precision at modest latency cost
7. **Optimization**: Adaptive retrieval parameters, parallel stages, early stopping achieve <500ms P99 latency
8. **Production**: Versioning, monitoring, fallbacks, and graceful degradation are essential

**Key metrics to track:**
- Retrieval recall@k (should be >80% for production)
- End-to-end latency P95/P99 (target <500ms for enterprise, <100ms for mobile)
- Reranking precision gain (should be 10-30% improvement)
- Index staleness (maximum age of newest indexed document)

**Next steps**: Deploy a baseline RAG system, measure actual latencies, and iteratively optimize the slowest stage. Typically, retrieval is fast (<50ms), reranking takes 50-150ms, and generation dominates at 200-400ms for long responses.
