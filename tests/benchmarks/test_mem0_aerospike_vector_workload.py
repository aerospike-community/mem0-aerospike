from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[3] / "ai-ecosystem-benchmark" / "src"))

from ai_ecosystem_benchmark import BenchmarkRunner  # noqa: E402

from benchmarks.mem0_aerospike_vector_workload import (  # noqa: E402
    DEFAULT_AEROSPIKE_CONNECTION_STRING,
    Mem0AerospikeVectorWorkload,
    _parse_connection_string,
    _record_payload,
)
import os  # noqa: E402

from benchmarks.mem0_redis_vector_workload import (  # noqa: E402
    WorkloadConfig,
    corpus_id,
    generate_payload_text,
    user_id_for_index,
)

AEROSPIKE_CONNECTION_STRING = os.environ.get(
    "AEROSPIKE_CONNECTION_STRING", DEFAULT_AEROSPIKE_CONNECTION_STRING
)


def _matches_filters(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    return all(payload.get(key) == value for key, value in filters.items())


class MemoryResult:
    """Minimal result object matching ``mem0.vector_stores.aerospike.MemoryResult``."""

    def __init__(self, id: str, payload: dict, score: float | None = None):
        self.id = id
        self.payload = payload
        self.score = score


class FakeAerospikeDB:
    """In-memory double for ``mem0.vector_stores.aerospike.AerospikeDB``'s public interface.

    Mirrors the provider's method signatures closely enough to exercise the
    workload's logic (corpus seeding, isolation, filtering) without a real
    Aerospike cluster.
    """

    def __init__(
        self,
        namespace: str,
        collection_name: str,
        embedding_model_dims: int,
        host: str = "localhost",
        port: int = 3000,
        **kwargs: Any,
    ) -> None:
        self.namespace = namespace
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.host = host
        self.port = port
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


def _make_workload(config: WorkloadConfig | None = None) -> tuple[Mem0AerospikeVectorWorkload, FakeAerospikeDB]:
    fake = FakeAerospikeDB(
        namespace="test",
        collection_name="mem0-aerospike-vector-benchmark-fake",
        embedding_model_dims=8,
    )
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
    workload = Mem0AerospikeVectorWorkload(
        config,
        aerospike_connection_string="localhost:3000:test",
        aerospike_db=fake,
    )
    return workload, fake


# ---------------------------------------------------------------------------
# Connection-string parsing
# ---------------------------------------------------------------------------


def test_parse_connection_string_splits_host_port_namespace() -> None:
    assert _parse_connection_string("127.0.0.1:3000:test") == ("127.0.0.1", 3000, "test")


def test_parse_connection_string_rejects_missing_components() -> None:
    with pytest.raises(ValueError, match="host:port:namespace"):
        _parse_connection_string("localhost:3000")


# ---------------------------------------------------------------------------
# Payload shape matches the Redis workload for comparability
# ---------------------------------------------------------------------------


def test_record_payload_matches_redis_workload_shape() -> None:
    payload = _record_payload(seed=42, index=7, payload_bytes=64, key_pool_size=4)

    assert set(payload) == {"data", "user_id"}
    assert len(payload["data"].encode()) == 64
    assert payload["user_id"] == user_id_for_index(7, 4)
    assert payload["data"] == generate_payload_text(42, 7, 64)


# ---------------------------------------------------------------------------
# Workload lifecycle and isolation (reusing Redis WorkloadConfig tests)
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

    workload.aerospike_insert()
    workload.aerospike_insert()
    assert len(fake.records) == baseline_count + 2

    workload.between_benchmarks()

    assert len(fake.records) == baseline_count


def test_aerospike_update_does_not_change_corpus_membership_or_user_id() -> None:
    workload, fake = _make_workload()
    workload.setup()
    baseline_user_ids = {vector_id: record["payload"]["user_id"] for vector_id, record in fake.records.items()}

    for _ in range(workload.config.corpus_size * 2):
        workload.aerospike_update()

    assert set(fake.records.keys()) == set(baseline_user_ids.keys())
    for vector_id, record in fake.records.items():
        assert record["payload"]["user_id"] == baseline_user_ids[vector_id]


def test_aerospike_insert_ids_never_collide_with_read_corpus_ids() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.aerospike_insert()

    corpus_ids = {corpus_id(i) for i in range(workload.config.corpus_size)}
    inserted_ids = set(fake.records.keys()) - corpus_ids
    assert len(inserted_ids) == 1


# ---------------------------------------------------------------------------
# Scenarios call the public interface correctly
# ---------------------------------------------------------------------------


def test_aerospike_search_calls_public_interface_with_user_id_filter() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.aerospike_search()

    assert len(fake.search_calls) == 1
    call = fake.search_calls[0]
    assert call["top_k"] == workload.config.top_k
    assert set(call["filters"].keys()) == {"user_id"}


def test_aerospike_list_calls_public_interface_with_user_id_filter() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.aerospike_list()

    assert len(fake.list_calls) == 1
    call = fake.list_calls[0]
    assert call["top_k"] == workload.config.top_k
    assert set(call["filters"].keys()) == {"user_id"}


def test_aerospike_insert_inserts_exactly_one_record_per_call() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.aerospike_insert()

    assert len(fake.insert_calls[-1]) == 1


def test_aerospike_update_replaces_vector_and_payload_on_existing_record() -> None:
    workload, fake = _make_workload()
    workload.setup()
    vector_id = corpus_id(0)
    original_vector = list(fake.records[vector_id]["vector"])

    workload.aerospike_update()

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
    assert DEFAULT_AEROSPIKE_CONNECTION_STRING not in serialized
    assert "aerospike_connection_string" not in serialized
    assert "password" not in serialized.lower()


# ---------------------------------------------------------------------------
# Local Aerospike smoke test (skipped if no local instance is reachable)
# ---------------------------------------------------------------------------


def _parse_host_port(connection_string: str) -> tuple[str, int]:
    host, port_str, _namespace = _parse_connection_string(connection_string)
    return host, int(port_str)


def _tcp_reachable(connection_string: str, timeout: float = 1.0) -> bool:
    host, port = _parse_host_port(connection_string)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _tcp_reachable(AEROSPIKE_CONNECTION_STRING),
    reason="no local Aerospike reachable on the configured connection string",
)
def test_smoke_all_scenarios_succeed_against_local_aerospike() -> None:
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
        collection_name="mem0-aerospike-vector-benchmark-smoke",
    )
    workload = Mem0AerospikeVectorWorkload(
        config,
        aerospike_connection_string=AEROSPIKE_CONNECTION_STRING,
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
    aerospike_results = results["backends"]["aerospike"]
    for scenario in (
        "aerospike_get",
        "aerospike_insert",
        "aerospike_list",
        "aerospike_search",
        "aerospike_update",
    ):
        assert aerospike_results[scenario]["successful_calls"] > 0
        assert aerospike_results[scenario]["failures"] == 0
