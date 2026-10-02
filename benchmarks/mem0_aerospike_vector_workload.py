"""mem0 Aerospike vector-store benchmark workload.

Benchmarks ``mem0.vector_stores.aerospike.AerospikeDB`` through its public
``VectorStoreBase`` interface, for direct comparison with
``mem0.vector_stores.redis.RedisDB`` in ``mem0_redis_vector_workload.py``.
Aerospike's native Top-K search performs an exact cosine-similarity ranking
over the candidate set (brute-force, no ANN), matching Redis's ``FLAT`` index.

Scenario discovery and isolation
---------------------------------
``ai-ecosystem-benchmark``'s ``BenchmarkRunner`` discovers benchmark methods via
``inspect.getmembers`` in alphabetical order:
``aerospike_get < aerospike_insert < aerospike_list < aerospike_search < aerospike_update``.
This workload seeds a single shared **read corpus** in ``setup()`` and no scenario
removes records from it, so discovery order can never deplete or empty another
scenario's data:

- ``aerospike_get``, ``aerospike_search``, ``aerospike_list`` are read-only against
  the read corpus.
- ``aerospike_update`` overwrites an existing corpus record's vector and payload
  in place, explicitly re-supplying its original ``user_id`` -- record count
  and ``user_id`` assignment never change, so the corpus stays valid for every
  other scenario regardless of run order.
- ``aerospike_insert`` adds new records with ids outside the read-corpus id range;
  ``between_benchmarks()`` removes them afterward so ``aerospike_insert`` always
  measures inserting into a same-size collection, and scenario ordering never
  changes measured insert cost.

``aerospike_delete`` is intentionally NOT implemented here. Benchmarking delete
throughput was scoped out because it is the only operation that would shrink the
corpus, which would otherwise require a dedicated pre-seeded delete pool and
pool-exhaustion accounting.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai_ecosystem_benchmark import BaseBenchmarkWorkload, BenchmarkRunner

from benchmarks.mem0_redis_vector_workload import (
    DISTANCE_METRIC,
    INDEX_ALGORITHM,
    WorkloadConfig,
    corpus_id,
    generate_payload_text,
    generate_vector,
    user_id_for_index,
)

DEFAULT_AEROSPIKE_CONNECTION_STRING = "localhost:3000:test"


def _parse_connection_string(connection_string: str) -> tuple[str, int, str]:
    """Parse ``host:port:namespace`` into its components."""
    parts = connection_string.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Aerospike connection string must be in the form host:port:namespace, got: {connection_string!r}"
        )
    host, port_str, namespace = parts
    return host, int(port_str), namespace


def _record_payload(seed: int, index: int, payload_bytes: int, key_pool_size: int) -> dict[str, str]:
    """Generate the same payload shape used by the Redis workload for comparability."""
    return {
        "data": generate_payload_text(seed, index, payload_bytes),
        "user_id": user_id_for_index(index, key_pool_size),
    }


class Mem0AerospikeVectorWorkload(BaseBenchmarkWorkload):
    """Benchmarks ``AerospikeDB`` insert/get/search/list/update through mem0's public interface."""

    def __init__(
        self,
        config: WorkloadConfig,
        *,
        aerospike_connection_string: str | None,
        aerospike_db: Any | None = None,
    ) -> None:
        super().__init__(aerospike_connection_string=aerospike_connection_string)
        self.config = config
        self._aerospike_db = aerospike_db
        self._read_counter = itertools.count()
        self._insert_counter = itertools.count(config.corpus_size)
        self._inserted_ids: list[str] = []

    def _ensure_db(self) -> Any:
        if self._aerospike_db is None:
            assert self.aerospike_connection_string is not None, (
                "aerospike_connection_string must be set to construct AerospikeDB"
            )
            from mem0.vector_stores.aerospike import AerospikeDB

            host, port, namespace = _parse_connection_string(self.aerospike_connection_string)
            self._aerospike_db = AerospikeDB(
                namespace=namespace,
                collection_name=self.config.collection_name,
                embedding_model_dims=self.config.embedding_model_dims,
                host=host,
                port=port,
            )
        return self._aerospike_db

    def setup(self) -> None:
        db = self._ensure_db()
        # reset() drops all records and recreates required secondary indexes, so a
        # prior failed or interrupted run cannot leak state into this one.
        db.reset()
        self._seed_corpus(db)
        self._read_counter = itertools.count()
        self._insert_counter = itertools.count(self.config.corpus_size)
        self._inserted_ids = []

    def _seed_corpus(self, db: Any) -> None:
        # The preview Aerospike SDK rejects a single batch beyond its buffer size,
        # so seed the corpus in fixed-size chunks while still issuing one logical insert.
        chunk_size = 1000
        for chunk_start in range(0, self.config.corpus_size, chunk_size):
            vectors = []
            payloads = []
            ids = []
            for index in range(chunk_start, min(chunk_start + chunk_size, self.config.corpus_size)):
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

    def aerospike_get(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        db.get(corpus_id(index))

    def aerospike_insert(self) -> None:
        db = self._ensure_db()
        index = next(self._insert_counter)
        vector = generate_vector(self.config.seed, index, self.config.embedding_model_dims)
        payload = _record_payload(
            self.config.seed,
            index,
            self.config.payload_bytes,
            self.config.key_pool_size,
        )
        vector_id = f"insert-{index}"
        db.insert(vectors=[vector], payloads=[payload], ids=[vector_id])
        self._inserted_ids.append(vector_id)

    def aerospike_list(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        user_id = user_id_for_index(index, self.config.key_pool_size)
        db.list(filters={"user_id": user_id}, top_k=self.config.top_k)

    def aerospike_search(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        query_vector = generate_vector(self.config.seed, index, self.config.embedding_model_dims)
        user_id = user_id_for_index(index, self.config.key_pool_size)
        db.search("", query_vector, top_k=self.config.top_k, filters={"user_id": user_id})

    def aerospike_update(self) -> None:
        db = self._ensure_db()
        index = next(self._read_counter) % self.config.corpus_size
        vector_id = corpus_id(index)
        # A different seed offset produces different content than the original
        # corpus record, while the user_id below re-supplies the same scoping
        # identity, keeping (id, user_id) unchanged.
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
    parser = argparse.ArgumentParser(description="Benchmark mem0's Aerospike vector store")
    parser.add_argument(
        "--aerospike-connection-string",
        default=DEFAULT_AEROSPIKE_CONNECTION_STRING,
        help="Aerospike seed in host:port:namespace form (default: localhost:3000:test)",
    )
    parser.add_argument("--collection-name", default="mem0-aerospike-vector-benchmark")
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
    parser.add_argument("--output", type=Path, default=Path("mem0-aerospike-vector-benchmark.json"))
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
    workload = Mem0AerospikeVectorWorkload(
        config,
        aerospike_connection_string=args.aerospike_connection_string,
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
