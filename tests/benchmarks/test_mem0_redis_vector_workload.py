from __future__ import annotations

import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[3] / "ai-ecosystem-benchmark" / "src"))

from ai_ecosystem_benchmark import BenchmarkRunner  # noqa: E402

from benchmarks.mem0_redis_vector_workload import (  # noqa: E402
    Mem0RedisVectorWorkload,
    WorkloadConfig,
    corpus_id,
    generate_payload_text,
    generate_vector,
    user_id_for_index,
)


def _matches_filters(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    return all(payload.get(key) == value for key, value in filters.items())


class FakeRedisDB:
    """In-memory double for mem0.vector_stores.redis.RedisDB's public interface.

    Mirrors RedisDB's method signatures closely enough to exercise the
    workload's logic (corpus seeding, isolation, filtering) without a real
    Redis Stack instance, matching the pattern used by
    agent-squad-aerospike's FakeStorage for AerospikeChatStorage/DynamoDbChatStorage.
    """

    def __init__(self) -> None:
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
        return SimpleNamespace(id=vector_id, payload=dict(record["payload"]), score=None)

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
            SimpleNamespace(id=vid, payload=dict(rec["payload"]), score=1.0)
            for vid, rec in self.records.items()
            if _matches_filters(rec["payload"], filters)
        ]
        return matches[:top_k]

    def list(self, filters=None, top_k=None):
        self.list_calls.append({"filters": filters, "top_k": top_k})
        matches = [
            SimpleNamespace(id=vid, payload=dict(rec["payload"]))
            for vid, rec in self.records.items()
            if _matches_filters(rec["payload"], filters)
        ]
        if top_k is not None:
            matches = matches[:top_k]
        return [matches]


def _make_workload(config: WorkloadConfig | None = None) -> tuple[Mem0RedisVectorWorkload, FakeRedisDB]:
    fake = FakeRedisDB()
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
    workload = Mem0RedisVectorWorkload(config, redis_url="redis://localhost:6379", redis_db=fake)
    return workload, fake


# ---------------------------------------------------------------------------
# WorkloadConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"embedding_model_dims": 0},
        {"corpus_size": 0},
        {"key_pool_size": 0},
        {"payload_bytes": 0},
        {"top_k": 0},
        {"qps": 0},
        {"duration_seconds": 0},
    ],
)
def test_workload_config_rejects_non_positive_dimensions(override: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        WorkloadConfig(**override)


def test_workload_config_rejects_negative_warmup() -> None:
    with pytest.raises(ValueError, match="warmup_seconds"):
        WorkloadConfig(warmup_seconds=-1)


def test_workload_config_rejects_key_pool_larger_than_corpus() -> None:
    with pytest.raises(ValueError, match="key_pool_size"):
        WorkloadConfig(corpus_size=10, key_pool_size=11)


def test_workload_config_accepts_valid_dimensions() -> None:
    config = WorkloadConfig(corpus_size=10, key_pool_size=10)
    assert config.corpus_size == 10


# ---------------------------------------------------------------------------
# Deterministic generation
# ---------------------------------------------------------------------------


def test_generate_vector_is_deterministic_and_unit_norm() -> None:
    first = generate_vector(seed=42, index=3, dims=16)
    second = generate_vector(seed=42, index=3, dims=16)

    assert first == second
    assert len(first) == 16
    norm = sum(component**2 for component in first) ** 0.5
    assert norm == pytest.approx(1.0)


def test_generate_vector_differs_across_index() -> None:
    assert generate_vector(seed=42, index=0, dims=16) != generate_vector(seed=42, index=1, dims=16)


def test_generate_payload_text_is_exact_size_and_deterministic() -> None:
    text = generate_payload_text(seed=42, index=5, size=64)

    assert len(text.encode()) == 64
    assert text == generate_payload_text(seed=42, index=5, size=64)


def test_generate_payload_text_rejects_size_too_small_for_prefix() -> None:
    with pytest.raises(ValueError, match="too small"):
        generate_payload_text(seed=42, index=5, size=1)


def test_user_id_for_index_round_robins_over_key_pool() -> None:
    assert user_id_for_index(0, key_pool_size=4) == user_id_for_index(4, key_pool_size=4)
    assert user_id_for_index(0, key_pool_size=4) != user_id_for_index(1, key_pool_size=4)


def test_corpus_generation_is_deterministic_across_independent_seedings() -> None:
    """Same seed + config produces identical vectors, user_id assignment, and payloads."""
    workload_a, fake_a = _make_workload()
    workload_b, fake_b = _make_workload()

    workload_a.setup()
    workload_b.setup()

    assert fake_a.records.keys() == fake_b.records.keys()
    for vector_id in fake_a.records:
        assert fake_a.records[vector_id]["vector"] == fake_b.records[vector_id]["vector"]
        assert fake_a.records[vector_id]["payload"] == fake_b.records[vector_id]["payload"]


# ---------------------------------------------------------------------------
# Deterministic per-call sampling
# ---------------------------------------------------------------------------


def test_per_call_sampling_sequence_is_deterministic() -> None:
    workload_a, fake_a = _make_workload()
    workload_b, fake_b = _make_workload()
    workload_a.setup()
    workload_b.setup()

    for _ in range(7):
        workload_a.redis_get()
        workload_b.redis_get()
        workload_a.redis_search()
        workload_b.redis_search()

    assert fake_a.get_calls == fake_b.get_calls
    assert [call["filters"] for call in fake_a.search_calls] == [
        call["filters"] for call in fake_b.search_calls
    ]
    assert [call["vectors"] for call in fake_a.search_calls] == [
        call["vectors"] for call in fake_b.search_calls
    ]


# ---------------------------------------------------------------------------
# setup() / between_benchmarks() lifecycle and isolation
# ---------------------------------------------------------------------------


def test_setup_truncates_and_fully_reseeds_the_read_corpus() -> None:
    workload, fake = _make_workload()
    # Simulate leftover state from a previous, possibly-interrupted run.
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

    workload.redis_insert()
    workload.redis_insert()
    assert len(fake.records) == baseline_count + 2

    workload.between_benchmarks()

    assert len(fake.records) == baseline_count


def test_redis_update_does_not_change_corpus_membership_or_user_id() -> None:
    """redis_update run before get/search/list must leave the corpus fully intact."""
    workload, fake = _make_workload()
    workload.setup()
    baseline_user_ids = {
        vector_id: record["payload"]["user_id"] for vector_id, record in fake.records.items()
    }

    for _ in range(workload.config.corpus_size * 2):
        workload.redis_update()

    assert set(fake.records.keys()) == set(baseline_user_ids.keys())
    for vector_id, record in fake.records.items():
        assert record["payload"]["user_id"] == baseline_user_ids[vector_id]


def test_redis_insert_ids_never_collide_with_read_corpus_ids() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.redis_insert()

    corpus_ids = {corpus_id(i) for i in range(workload.config.corpus_size)}
    inserted_ids = set(fake.records.keys()) - corpus_ids
    assert len(inserted_ids) == 1


# ---------------------------------------------------------------------------
# Scenarios call the public interface correctly
# ---------------------------------------------------------------------------


def test_redis_search_calls_public_interface_with_user_id_filter() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.redis_search()

    assert len(fake.search_calls) == 1
    call = fake.search_calls[0]
    assert call["top_k"] == workload.config.top_k
    assert set(call["filters"].keys()) == {"user_id"}


def test_redis_list_calls_public_interface_with_user_id_filter() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.redis_list()

    assert len(fake.list_calls) == 1
    call = fake.list_calls[0]
    assert call["top_k"] == workload.config.top_k
    assert set(call["filters"].keys()) == {"user_id"}


def test_redis_insert_inserts_exactly_one_record_per_call() -> None:
    workload, fake = _make_workload()
    workload.setup()

    workload.redis_insert()

    assert len(fake.insert_calls[-1]) == 1


def test_redis_update_replaces_vector_and_payload_on_existing_record() -> None:
    workload, fake = _make_workload()
    workload.setup()
    vector_id = corpus_id(0)
    original_vector = list(fake.records[vector_id]["vector"])

    workload.redis_update()

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
    assert "redis://localhost:6379" not in serialized
    assert "redis_url" not in serialized
    assert "password" not in serialized.lower()


# ---------------------------------------------------------------------------
# Local Redis Stack smoke test (skipped if no local instance is reachable)
# ---------------------------------------------------------------------------


def _tcp_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _tcp_reachable("localhost", 6379), reason="no local Redis Stack reachable on :6379"
)
def test_smoke_all_scenarios_succeed_against_local_redis_stack() -> None:
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
        collection_name="mem0-redis-vector-benchmark-smoke",
    )
    workload = Mem0RedisVectorWorkload(config, redis_url="redis://localhost:6379")
    runner = BenchmarkRunner(
        queries_per_second=config.qps,
        scheduler_thread_count=1,
        worker_thread_count=8,
        runtime_per_function=config.duration_seconds,
        workload=workload,
    )

    runner.run()

    results = runner.results()
    redis_results = results["backends"]["redis"]
    for scenario in (
        "redis_get",
        "redis_insert",
        "redis_list",
        "redis_search",
        "redis_update",
    ):
        assert redis_results[scenario]["successful_calls"] > 0
        assert redis_results[scenario]["failures"] == 0
