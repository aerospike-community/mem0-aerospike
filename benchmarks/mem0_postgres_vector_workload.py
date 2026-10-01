"""mem0 pgvector (PostgreSQL) vector-store benchmark workload.

Benchmarks ``mem0.vector_stores.pgvector.PGVector`` through its public
``VectorStoreBase`` interface, for direct comparison with Redis and Aerospike.
By default ``PGVector`` is created with ``hnsw=False`` and ``diskann=False``,
so its ``search()`` performs an exact distance-ordering over the filtered
candidate set — the same brute-force shape as Redis ``FLAT`` and Aerospike
exact Top-K.

Scenario discovery and isolation follow the same pattern as the Redis and
Aerospike workloads: a shared read corpus is seeded in ``setup()``,
``between_benchmarks()`` trims only records inserted beyond the corpus, and
``postgres_update`` keeps each record's ``user_id`` unchanged.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai_ecosystem_benchmark import BaseBenchmarkWorkload, BenchmarkRunner

from benchmarks.mem0_redis_vector_workload import (
    DISTANCE_METRIC,
    INDEX_ALGORITHM,
    WorkloadConfig,
    generate_payload_text,
    generate_vector,
    user_id_for_index,
)

DEFAULT_POSTGRES_CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5432/postgres"
_ID_NAMESPACE = uuid.NAMESPACE_DNS


def _record_id(prefix: str, index: int) -> str:
    """Deterministic UUID string that satisfies PGVector's UUID primary key."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{prefix}-{index}"))


def corpus_id(index: int) -> str:
    """The read-corpus record id for a given corpus index."""
    return _record_id("corpus", index)


def insert_id(index: int) -> str:
    """Id for a record inserted beyond the read corpus."""
    return _record_id("insert", index)


def _record_payload(seed: int, index: int, payload_bytes: int, key_pool_size: int) -> dict[str, str]:
    """Generate the same payload shape used by the Redis workload for comparability."""
    return {
        "data": generate_payload_text(seed, index, payload_bytes),
        "user_id": user_id_for_index(index, key_pool_size),
    }


class Mem0PostgresVectorWorkload(BaseBenchmarkWorkload):
    """Benchmarks ``PGVector`` insert/get/search/list/update through mem0's public interface."""

    def __init__(
        self,
        config: WorkloadConfig,
        *,
        postgres_connection_string: str | None,
        postgres_db: Any | None = None,
    ) -> None:
        super().__init__(postgres_connection_string=postgres_connection_string)
        self.config = config
        self._postgres_db = postgres_db
        self._read_counter = itertools.count()
        self._insert_counter = itertools.count(config.corpus_size)
        self._inserted_ids: list[str] = []

    def _ensure_db(self) -> Any:
        if self._postgres_db is None:
            assert self.postgres_connection_string is not None, (
                "postgres_connection_string must be set to construct PGVector"
            )
            from mem0.vector_stores.pgvector import PGVector

            self._postgres_db = PGVector(
                dbname="postgres",
                collection_name=self.config.collection_name,
                embedding_model_dims=self.config.embedding_model_dims,
                user="",
                password="",
                host="",
                port=0,
                diskann=False,
                hnsw=False,
                minconn=2,
                maxconn=64,
                connection_string=self.postgres_connection_string,
            )
        return self._postgres_db

    def setup(self) -> None:
        db = self._ensure_db()
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
            vectors.append(generate_vector(self.config.seed, index, self.config.embedding_model_dims))
            payloads.append(
                _record_payload(
                    self.config.seed,
                    index,
                    self.config.payload_bytes,
                    self.config.key_pool_size,
                )
            )
            ids.append(corpus_id(index))
        db.insert(vectors=vectors, payloads=payloads, ids=ids)

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

    def postgres_get(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        db.get(corpus_id(index))

    def postgres_insert(self) -> None:
        db = self._ensure_db()
        index = next(self._insert_counter)
        vector = generate_vector(self.config.seed, index, self.config.embedding_model_dims)
        payload = _record_payload(
            self.config.seed,
            index,
            self.config.payload_bytes,
            self.config.key_pool_size,
        )
        vector_id = insert_id(index)
        db.insert(vectors=[vector], payloads=[payload], ids=[vector_id])
        self._inserted_ids.append(vector_id)

    def postgres_list(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        user_id = user_id_for_index(index, self.config.key_pool_size)
        db.list(filters={"user_id": user_id}, top_k=self.config.top_k)

    def postgres_search(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        query_vector = generate_vector(self.config.seed, index, self.config.embedding_model_dims)
        user_id = user_id_for_index(index, self.config.key_pool_size)
        db.search("", query_vector, top_k=self.config.top_k, filters={"user_id": user_id})

    def postgres_update(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        vector_id = corpus_id(index)
        new_vector = generate_vector(self.config.seed + 1, index, self.config.embedding_model_dims)
        new_payload = {
            "data": _record_payload(
                self.config.seed + 1,
                index,
                self.config.payload_bytes,
                self.config.key_pool_size,
            )["data"],
            "user_id": user_id_for_index(index, self.config.key_pool_size),
        }
        db.update(vector_id=vector_id, vector=new_vector, payload=new_payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark mem0's pgvector store")
    parser.add_argument(
        "--postgres-connection-string",
        default=DEFAULT_POSTGRES_CONNECTION_STRING,
        help="PostgreSQL connection URI (default: postgresql://postgres:postgres@localhost:5432/postgres)",
    )
    parser.add_argument("--collection-name", default="mem0-postgres-vector-benchmark")
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
    parser.add_argument("--output", type=Path, default=Path("mem0-postgres-vector-benchmark.json"))
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
    workload = Mem0PostgresVectorWorkload(
        config,
        postgres_connection_string=args.postgres_connection_string,
    )
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
