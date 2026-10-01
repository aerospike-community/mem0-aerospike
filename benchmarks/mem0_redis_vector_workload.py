"""mem0 Redis vector-store benchmark workload.

Benchmarks ``mem0.vector_stores.redis.RedisDB`` through its public
``VectorStoreBase`` interface (never through raw ``redisvl``/``redis-py`` calls),
for later comparison against the in-progress Aerospike vector store from the
``add-aerospike-vector-store`` change. Redis's ``FLAT`` index (exact, brute-force
cosine similarity) is the correct comparator for Aerospike's exact-Top-K search --
this workload never configures ``HNSW``.

Scenario discovery and isolation
---------------------------------
``ai-ecosystem-benchmark``'s ``BenchmarkRunner`` discovers benchmark methods via
``inspect.getmembers``, which returns them in alphabetical order:
``redis_get < redis_insert < redis_list < redis_search < redis_update``. This
workload seeds a single shared **read corpus** in ``setup()`` and no scenario
removes records from it, so that discovery order (or any other run order) can
never deplete or empty another scenario's data:

- ``redis_get``, ``redis_search``, ``redis_list`` are read-only against the
  read corpus.
- ``redis_update`` overwrites an existing corpus record's vector and payload
  in place, explicitly re-supplying its original ``user_id`` -- record count
  and ``user_id`` assignment never change, so the corpus stays valid for every
  other scenario regardless of run order.
- ``redis_insert`` adds new records with ids outside the read-corpus id range;
  ``between_benchmarks()`` removes them afterward so ``redis_insert`` always
  measures inserting into a same-size collection, and scenario ordering never
  changes measured insert cost.

``redis_delete`` is intentionally NOT implemented here. Benchmarking delete
throughput was scoped out (see ``design.md``'s Non-Goals) specifically because
it is the only operation that would shrink the corpus, which would otherwise
require a dedicated pre-seeded delete pool and pool-exhaustion accounting. Do
not add a ``redis_delete`` scenario without re-reading that rationale and
reintroducing an isolated pool for it.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ai_ecosystem_benchmark import BaseBenchmarkWorkload, BenchmarkRunner

INDEX_ALGORITHM = "FLAT"
DISTANCE_METRIC = "cosine"


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    """Reproducible workload dimensions for the mem0 Redis vector benchmark."""

    seed: int = 20260923
    embedding_model_dims: int = 1536
    corpus_size: int = 1000
    key_pool_size: int = 50
    payload_bytes: int = 256
    top_k: int = 10
    qps: int = 50
    warmup_seconds: int = 1
    duration_seconds: int = 10
    collection_name: str = "mem0-redis-vector-benchmark"

    def __post_init__(self) -> None:
        positive = {
            "embedding_model_dims": self.embedding_model_dims,
            "corpus_size": self.corpus_size,
            "key_pool_size": self.key_pool_size,
            "payload_bytes": self.payload_bytes,
            "top_k": self.top_k,
            "qps": self.qps,
            "duration_seconds": self.duration_seconds,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("workload dimensions must be positive")
        if self.warmup_seconds < 0:
            raise ValueError("warmup_seconds must not be negative")
        if self.key_pool_size > self.corpus_size:
            raise ValueError("key_pool_size must not exceed corpus_size")


def corpus_id(index: int) -> str:
    """The read-corpus record id for a given corpus index."""
    return f"corpus-{index}"


def user_id_for_index(index: int, key_pool_size: int) -> str:
    """Deterministic round-robin ``user_id`` assignment for a corpus index."""
    return f"user-{index % key_pool_size}"


def generate_vector(seed: int, index: int, dims: int) -> list[float]:
    """Deterministically generate a unit-norm embedding for record ``index``.

    Seeded per-index (``seed + index``) rather than by advancing a single
    shared generator, so the same ``(seed, index, dims)`` always produces the
    same vector regardless of generation order.
    """
    rng = np.random.default_rng(seed + index)
    vector = rng.standard_normal(dims).astype(np.float64)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def generate_payload_text(seed: int, index: int, size: int) -> str:
    """Deterministically generate exact-``size``-byte payload text for record ``index``."""
    prefix = f"{seed}:{index}:"
    if len(prefix.encode()) > size:
        raise ValueError("payload size is too small for deterministic prefix")
    return prefix + "x" * (size - len(prefix.encode()))


class Mem0RedisVectorWorkload(BaseBenchmarkWorkload):
    """Benchmarks ``RedisDB`` insert/get/search/list/update through mem0's public interface."""

    def __init__(
        self,
        config: WorkloadConfig,
        *,
        redis_url: str | None,
        redis_db: Any | None = None,
    ) -> None:
        super().__init__(redis_connection_string=redis_url)
        self.config = config
        self._redis_db = redis_db
        self._read_counter = itertools.count()
        self._insert_counter = itertools.count(config.corpus_size)
        self._inserted_ids: list[str] = []

    def _ensure_db(self) -> Any:
        if self._redis_db is None:
            assert self.redis_connection_string is not None, (
                "redis_url must be set to construct RedisDB"
            )
            from mem0.vector_stores.redis import RedisDB

            self._redis_db = RedisDB(
                redis_url=self.redis_connection_string,
                collection_name=self.config.collection_name,
                embedding_model_dims=self.config.embedding_model_dims,
            )
        return self._redis_db

    def setup(self) -> None:
        db = self._ensure_db()
        # reset() drops all keys and recreates the index, so a prior failed or
        # interrupted run cannot leak state into this one.
        db.reset()
        self._seed_corpus(db)
        self._read_counter = itertools.count()
        self._insert_counter = itertools.count(self.config.corpus_size)
        self._inserted_ids = []

    def _seed_corpus(self, db: Any) -> None:
        vectors = []
        payloads = []
        ids = []
        for index in range(self.config.corpus_size):
            vectors.append(
                generate_vector(self.config.seed, index, self.config.embedding_model_dims)
            )
            payloads.append(self._record_payload(self.config.seed, index))
            ids.append(corpus_id(index))
        db.insert(vectors=vectors, payloads=payloads, ids=ids)

    def _record_payload(self, seed: int, index: int) -> dict[str, str]:
        return {
            "data": generate_payload_text(seed, index, self.config.payload_bytes),
            "user_id": user_id_for_index(index, self.config.key_pool_size),
        }

    def between_benchmarks(self) -> None:
        db = self._ensure_db()
        for vector_id in self._inserted_ids:
            db.delete(vector_id)
        self._inserted_ids = []

    def teardown(self) -> None:
        return None

    def benchmark_metadata(self) -> dict[str, object]:
        versions = {}
        for package in ("mem0ai", "ai-ecosystem-benchmark"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "source"
        return {
            **super().benchmark_metadata(),
            "environment": "local",
            "index_algorithm": INDEX_ALGORITHM,
            "distance_metric": DISTANCE_METRIC,
            "workload": asdict(self.config),
            "versions": versions,
        }

    # --- scenarios (discovered alphabetically by the runner) ---------------

    def redis_get(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        db.get(corpus_id(index))

    def redis_insert(self) -> None:
        db = self._ensure_db()
        index = next(self._insert_counter)
        vector = generate_vector(self.config.seed, index, self.config.embedding_model_dims)
        payload = self._record_payload(self.config.seed, index)
        vector_id = f"insert-{index}"
        db.insert(vectors=[vector], payloads=[payload], ids=[vector_id])
        self._inserted_ids.append(vector_id)

    def redis_list(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        user_id = user_id_for_index(index, self.config.key_pool_size)
        db.list(filters={"user_id": user_id}, top_k=self.config.top_k)

    def redis_search(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        query_vector = generate_vector(self.config.seed, index, self.config.embedding_model_dims)
        user_id = user_id_for_index(index, self.config.key_pool_size)
        db.search("", query_vector, top_k=self.config.top_k, filters={"user_id": user_id})

    def redis_update(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        vector_id = corpus_id(index)
        # A different seed offset produces different content than the original
        # corpus record, while user_id_for_index(index, ...) below re-supplies
        # the same user_id, keeping the record's isolation-relevant identity
        # (id, user_id) unchanged.
        new_vector = generate_vector(
            self.config.seed + 1, index, self.config.embedding_model_dims
        )
        new_payload = {
            "data": generate_payload_text(
                self.config.seed + 1, index, self.config.payload_bytes
            ),
            "user_id": user_id_for_index(index, self.config.key_pool_size),
        }
        db.update(vector_id=vector_id, vector=new_vector, payload=new_payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark mem0's Redis vector store")
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--collection-name", default="mem0-redis-vector-benchmark")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--embedding-dims", type=int, default=1536)
    parser.add_argument("--corpus-size", type=int, default=1000)
    parser.add_argument("--key-pool-size", type=int, default=50)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--qps", type=int, default=50)
    parser.add_argument("--scheduler-threads", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=256)
    parser.add_argument("--warmup-seconds", type=int, default=1)
    parser.add_argument("--duration-seconds", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("mem0-redis-vector-benchmark.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = WorkloadConfig(
        seed=args.seed,
        embedding_model_dims=args.embedding_dims,
        corpus_size=args.corpus_size,
        key_pool_size=args.key_pool_size,
        payload_bytes=args.payload_bytes,
        top_k=args.top_k,
        qps=args.qps,
        warmup_seconds=args.warmup_seconds,
        duration_seconds=args.duration_seconds,
        collection_name=args.collection_name,
    )
    workload = Mem0RedisVectorWorkload(config, redis_url=args.redis_url)
    runner = BenchmarkRunner(
        queries_per_second=config.qps,
        scheduler_thread_count=args.scheduler_threads,
        worker_thread_count=args.worker_threads,
        runtime_per_function=config.duration_seconds,
        workload=workload,
    )
    runner.run()
    runner.print_metrics()
    runner.write_json(args.output)
    print(json.dumps({"results": str(args.output)}, indent=2))
