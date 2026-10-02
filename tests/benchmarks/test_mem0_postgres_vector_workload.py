from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[3] / "ai-ecosystem-benchmark" / "src"))

from ai_ecosystem_benchmark import BenchmarkRunner  # noqa: E402

from benchmarks.mem0_postgres_vector_workload import (  # noqa: E402
    DEFAULT_POSTGRES_CONNECTION_STRING,
    Mem0PostgresVectorWorkload,
    corpus_id,
    insert_id,
    _record_payload,
)
from benchmarks.mem0_redis_vector_workload import (  # noqa: E402
    WorkloadConfig,
    generate_payload_text,
    user_id_for_index,
)


def _matches_filters(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    return all(payload.get(key) == value for key, value in filters.items())


class MemoryResult:
    """Minimal result object matching ``mem0.vector_stores.pgvector.OutputData``."""

    def __init__(self, id: str, payload: dict, score: float | None = None):
        self.id = id
        self.payload = payload
        self.score = score


class FakePGVector:
    """In-memory double for ``mem0.vector_stores.pgvector.PGVector``'s public interface."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.records: dict[str, dict[str, Any]] = {}
        self.reset_calls = 0
        self.get_calls: list[str] = []
        self.search_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.insert_calls: list[list[str]] = []

    def reset(self) -> None:
        self.records = {}
        self.reset_calls += 1

    def insert(self, vectors, payloads=None, ids=None) -> None:
        payloads = payloads or [{}] * len(vectors)
        ids = ids or [None] * len(vectors)
        for vector, payload, vector_id in zip(vectors, payloads, ids):
            self.records[vector_id] = {"vector": list(vector), "payload": dict(payload)}
        self.insert_calls.append(list(ids))

    def get(self, vector_id):
        self.get_calls.append(vector_id)
        record = self.records.get(vector_id)
        if record is None:
            return None
        return MemoryResult(id=vector_id, payload=dict(record["payload"]), score=None)

    def update(self, vector_id=None, vector=None, payload=None) -> None:
        self.update_calls.append({"vector_id": vector_id, "vector": vector, "payload": payload})
        record = self.records.setdefault(vector_id, {"vector": None, "payload": {}})
        if vector is not None:
            record["vector"] = list(vector)
        if payload is not None:
            record["payload"] = dict(payload)

    def delete(self, vector_id) -> None:
        self.records.pop(vector_id, None)

    def search(self, query, vectors, top_k=5, filters=None):
        self.search_calls.append({"vectors": vectors, "top_k": top_k, "filters": filters})
        matches = [
            MemoryResult(id=vid, payload=dict(rec["payload"]), score=1.0)
            for vid, rec in self.records.items()
            if _matches_filters(rec["payload"], filters)
        ]
        return matches[:top_k]

    def list(self, filters=None, top_k=None):
        self.list_calls.append({"filters": filters, "top_k": top_k})
        matches = [
            MemoryResult(id=vid, payload=dict(rec["payload"]))
            for vid, rec in self.records.items()
            if _matches_filters(rec["payload"], filters)
        ]
        if top_k is not None:
            matches = matches[:top_k]
        return [matches]


def _make_workload(config: WorkloadConfig | None = None) -> tuple[Mem0PostgresVectorWorkload, FakePGVector]:
    fake = FakePGVector()
    config = config or WorkloadConfig(
        seed=42,
        embedding_model_dims=8,
        corpus_size=20,
        key_pool_size=4,
        payload_bytes=32,
        top_k=5,
        qps=10,
        warmup_seconds=0,
        duration_seconds=1,
    )
    workload = Mem0PostgresVectorWorkload(
        config,
        postgres_connection_string=DEFAULT_POSTGRES_CONNECTION_STRING,
        postgres_db=fake,
    )
    return workload, fake


# ---------------------------------------------------------------------------
# UUID id generation and payload shape
# ---------------------------------------------------------------------------


def test_corpus_id_is_deterministic_uuid() -> None:
    first = corpus_id(7)
    second = corpus_id(7)
    assert first == second
    assert first != corpus_id(8)


def test_insert_id_does_not_collide_with_corpus_id() -> None:
    assert insert_id(0) != corpus_id(0)


def test_record_payload_matches_redis_workload_shape() -> None:
    payload = _record_payload(seed=42, index=7, payload_bytes=64, key_pool_size=4)

    assert set(payload) == {"data", "user_id"}
    assert len(payload["data"].encode()) == 64
    assert payload["user_id"] == user_id_for_index(7, 4)
    assert payload["data"] == generate_payload_text(42, 7, 64)


# ---------------------------------------------------------------------------
# Workload lifecycle and isolation
# ---------------------------------------------------------------------------


def test_setup_truncates_and_fully_reseeds_the_read_corpus() -> None:
    workload, fake = _make_workload()
    fake.records["stale-record"] = {"vector": [0.0], "payload": {"user_id": "ghost"}}

    workload.setup()

    assert fake.reset_calls == 1
    assert "stale-record" not in fake.records
    assert len(fake.records) == workload.config.corpus_size
    assert all(corpus_id(i) in fake.records for i in range(workload.config.corpus_size))


def test_between_benchmarks_trims_records_inserted_beyond_corpus_size() -> None:
    workload, fake = _make_workload()
    workload.setup()
    baseline_count = len(fake.records)

    workload.postgres_insert()
    workload.postgres_insert()
    assert len(fake.records) == baseline_count + 2

    workload.between_benchmarks()

    assert len(fake.records) == baseline_count


def test_postgres_update_does_not_change_corpus_membership_or_user_id() -> None:
    workload, fake = _make_workload()
    workload.setup()
    baseline_user_ids = {vector_id: record["payload"]["user_id"] for vector_id, record in fake.records.items()}

    for _ in range(workload.config.corpus_size * 2):
        workload.postgres_update()

    assert set(fake.records.keys()) == set(baseline_user_ids.keys())
    for vector_id, record in fake.records.items():
        assert record["payload"]["user_id"] == baseline_user_ids[vector_id]


def test_postgres_insert_ids_never_collide_with_read_corpus_ids() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.postgres_insert()

    corpus_ids = {corpus_id(i) for i in range(workload.config.corpus_size)}
    inserted_ids = set(fake.records.keys()) - corpus_ids
    assert len(inserted_ids) == 1


# ---------------------------------------------------------------------------
# Scenarios call the public interface correctly
# ---------------------------------------------------------------------------


def test_postgres_search_calls_public_interface_with_user_id_filter() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.postgres_search()

    assert len(fake.search_calls) == 1
    call = fake.search_calls[0]
    assert call["top_k"] == workload.config.top_k
    assert set(call["filters"].keys()) == {"user_id"}


def test_postgres_list_calls_public_interface_with_user_id_filter() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.postgres_list()

    assert len(fake.list_calls) == 1
    call = fake.list_calls[0]
    assert call["top_k"] == workload.config.top_k
    assert set(call["filters"].keys()) == {"user_id"}


def test_postgres_insert_inserts_exactly_one_record_per_call() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.postgres_insert()

    assert len(fake.insert_calls[-1]) == 1


def test_postgres_update_replaces_vector_and_payload_on_existing_record() -> None:
    workload, fake = _make_workload()
    workload.setup()
    vector_id = corpus_id(0)
    original_vector = list(fake.records[vector_id]["vector"])

    workload.postgres_update()

    assert fake.records[vector_id]["vector"] != original_vector


# ---------------------------------------------------------------------------
# Results and comparability metadata
# ---------------------------------------------------------------------------


def test_benchmark_metadata_includes_comparability_disclosures() -> None:
    workload, _ = _make_workload()

    metadata = workload.benchmark_metadata()

    assert metadata["environment"] == "local"
    assert metadata["index_algorithm"] == "FLAT"
    assert metadata["distance_metric"] == "cosine"
    assert metadata["workload"]["corpus_size"] == workload.config.corpus_size
    assert metadata["workload"]["embedding_model_dims"] == workload.config.embedding_model_dims
    assert "mem0ai" in metadata["versions"]
    assert "ai-ecosystem-benchmark" in metadata["versions"]


def test_benchmark_metadata_never_includes_connection_string_or_credentials() -> None:
    workload, _ = _make_workload()

    metadata = workload.benchmark_metadata()

    serialized = str(metadata)
    assert DEFAULT_POSTGRES_CONNECTION_STRING not in serialized
    assert "postgres_connection_string" not in serialized
    assert "password" not in serialized.lower()


# ---------------------------------------------------------------------------
# Local Postgres smoke test (skipped if no local instance is reachable)
# ---------------------------------------------------------------------------


def _tcp_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _parse_host_port(connection_string: str) -> tuple[str, int]:
    # Minimal parser for postgresql://user:pass@host:port/dbname
    rest = connection_string.split("://", 1)[-1]
    host_port = rest.split("@", 1)[-1].split("/", 1)[0]
    if ":" in host_port:
        host, port_str = host_port.rsplit(":", 1)
        return host, int(port_str)
    return host_port, 5432


@pytest.mark.skipif(
    not _tcp_reachable(*_parse_host_port(DEFAULT_POSTGRES_CONNECTION_STRING)),
    reason="no local Postgres reachable on the default connection string",
)
def test_smoke_all_scenarios_succeed_against_local_postgres() -> None:
    config = WorkloadConfig(
        seed=7,
        embedding_model_dims=16,
        corpus_size=50,
        key_pool_size=5,
        payload_bytes=32,
        top_k=5,
        qps=10,
        warmup_seconds=0,
        duration_seconds=1,
        collection_name="mem0-postgres-vector-benchmark-smoke",
    )
    workload = Mem0PostgresVectorWorkload(
        config,
        postgres_connection_string=DEFAULT_POSTGRES_CONNECTION_STRING,
    )
    runner = BenchmarkRunner(
        queries_per_second=config.qps,
        scheduler_thread_count=1,
        worker_thread_count=8,
        runtime_per_function=config.duration_seconds,
        workload=workload,
    )

    runner.run()

    results = runner.results()
    postgres_results = results["backends"]["postgres"]
    for scenario in (
        "postgres_get",
        "postgres_insert",
        "postgres_list",
        "postgres_search",
        "postgres_update",
    ):
        assert postgres_results[scenario]["successful_calls"] > 0
        assert postgres_results[scenario]["failures"] == 0
