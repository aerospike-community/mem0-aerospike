"""Integration tests for the Aerospike vector store provider.

Tests exercise the public ``VectorStoreBase`` seam implemented by ``AerospikeDB``.
A running Aerospike Server 8.1.3+ instance is required (default localhost:3000).
"""

import os
import time
import warnings
from datetime import datetime, timezone

import pytest

aerospike_sdk = pytest.importorskip("aerospike_sdk")

from aerospike_sdk import Behavior, DataSet  # noqa: E402
from aerospike_sdk.sync import ClusterDefinition  # noqa: E402

from mem0.vector_stores.aerospike import AerospikeDB  # noqa: E402

NAMESPACE = os.environ.get("AEROSPIKE_NAMESPACE", "test")
HOST = os.environ.get("AEROSPIKE_HOST", "localhost")
PORT = int(os.environ.get("AEROSPIKE_PORT", "3000"))
EMBEDDING_DIMS = 4


def _truncate_collection(db: AerospikeDB) -> None:
    """Best-effort truncate of the test set via the SDK."""
    try:
        with ClusterDefinition(HOST, PORT).connect() as cluster:
            session = cluster.create_session(Behavior.DEFAULT)
            session.truncate(DataSet.of(db.namespace, db.collection_name))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"truncate failed for {db.namespace}/{db.collection_name}: {exc}")


@pytest.fixture
def db():
    """AerospikeDB instance backed by a truncated, shared test set.

    The collection name is fixed (not per-test-random) so that `create_col()`
    reuses the same three secondary indexes across the whole session instead
    of creating fresh ones per test: index counts are capped server-side, and
    a suite-sized burst of brand-new indexes exhausts that cap quickly.
    `create_col()` is idempotent, so reuse is safe.
    """
    instance = AerospikeDB(
        namespace=NAMESPACE,
        collection_name="mem0_aerospike_test",
        embedding_model_dims=EMBEDDING_DIMS,
        host=HOST,
        port=PORT,
    )
    _truncate_collection(instance)
    yield instance
    _truncate_collection(instance)


@pytest.fixture
def db_allowing_scans():
    """Instance configured to allow scan-backed filtered queries."""
    instance = AerospikeDB(
        namespace=NAMESPACE,
        collection_name="mem0_aerospike_test_scans",
        embedding_model_dims=EMBEDDING_DIMS,
        host=HOST,
        port=PORT,
        allow_scans_with_where=True,
    )
    _truncate_collection(instance)
    yield instance
    _truncate_collection(instance)


def _iso(ts: datetime) -> str:
    # Millisecond precision: the provider stores created_at/updated_at as
    # created_at_ms/updated_at_ms int64 epoch-ms bins (see spec), so a
    # microsecond-precision input would not round-trip byte-for-byte.
    ts = ts.replace(microsecond=(ts.microsecond // 1000) * 1000, tzinfo=timezone.utc)
    return ts.isoformat(timespec="milliseconds")


def _sample_vector(offset: float = 1.0) -> list:
    base = [0.0] * EMBEDDING_DIMS
    base[0] = offset
    return base


def _payload(
    user_id: str,
    data: str = "sample memory",
    created_at: str = None,
    updated_at: str = None,
    **metadata,
) -> dict:
    now = _iso(datetime.now(timezone.utc))
    payload = {
        "data": data,
        "hash": f"hash-{user_id}-{data}",
        "user_id": user_id,
        "created_at": created_at or now,
        "updated_at": updated_at or now,
    }
    payload.update(metadata)
    return payload


# 2. Collection lifecycle


def test_resolve_scope_treats_eq_operator_as_scoped(db):
    """Scope fields are recognized as scoping predicates whether passed as a
    literal value or as an ``eq`` operator dict.

    Regression test: previously ``{"user_id": {"eq": "alice"}}`` was treated
    as unscoped, causing list()/search() to fall back to a full scan.
    """
    assert db._resolve_scope({"user_id": "alice"}) == ("user_id", "alice")
    assert db._resolve_scope({"user_id": {"eq": "alice"}}) == ("user_id", "alice")
    assert db._resolve_scope({"agent_id": {"eq": "bot"}}) == ("agent_id", "bot")
    assert db._resolve_scope({"run_id": {"eq": "session"}}) == ("run_id", "session")


def test_resolve_scope_non_eq_operator_is_not_indexable(db):
    """A non-equality operator on a scope field counts as scoped (no scan guard
    error) but does not produce an indexable equality value."""
    assert db._resolve_scope({"user_id": {"ne": "alice"}}) == (None, None)


def test_create_col_makes_collection_visible(db):
    """2.1 RED: a fresh AerospikeDB instance exposes its collection via list_cols."""
    collections = db.list_cols()
    assert db.collection_name in collections


def test_delete_col_waits_for_truncate_task(db, mocker):
    """2.2 RED: delete_col waits for the truncate background task before returning.

    Regression test: previously the return value of ``session.truncate()`` was
    ignored, so ``reset()`` could recreate indexes while truncation was still in
    progress.
    """
    mock_task = mocker.Mock()
    mocker.patch.object(db.session, "truncate", return_value=mock_task)

    db.delete_col()

    db.session.truncate.assert_called_once_with(db.dataset)
    mock_task.wait_till_complete_blocking.assert_called_once()


def test_delete_col_removes_all_records(db):
    """2.3 RED: delete_col removes every record from the set."""
    db.insert(
        vectors=[_sample_vector()],
        payloads=[_payload("delete_col_user")],
        ids=["record-to-delete"],
    )
    # Truncate's cutoff has ~1s resolution server-side: a truncate issued in
    # the same second as the write it should remove can race and miss it.
    time.sleep(1.1)
    db.delete_col()
    assert db.get("record-to-delete") is None


def test_reset_empties_the_set_and_leaves_it_usable(db):
    """8.1: reset() (delete_col + recreate indexes) empties the set and the
    collection remains fully usable afterward -- not covered by the delete_col
    cycle above, which only checks the delete half."""
    db.insert(
        vectors=[_sample_vector()],
        payloads=[_payload("reset_user")],
        ids=["record-before-reset"],
    )
    time.sleep(1.1)
    db.reset()
    assert db.get("record-before-reset") is None

    db.insert(
        vectors=[_sample_vector()],
        payloads=[_payload("reset_user")],
        ids=["record-after-reset"],
    )
    assert db.get("record-after-reset") is not None
    assert db.collection_name in db.list_cols()


def test_col_info_reports_total_record_count(db):
    """2.5 RED: col_info returns the cluster-wide record count."""
    vectors = [_sample_vector(i) for i in range(3)]
    payloads = [_payload(f"info_user_{i}") for i in range(3)]
    ids = [f"info-{i}" for i in range(3)]
    db.insert(vectors=vectors, payloads=payloads, ids=ids)

    info = db.col_info()
    assert info["count"] == 3


# 3. Insert, get, delete


def test_insert_and_get_roundtrip(db):
    """3.1 RED: a single insert/get returns the original payload."""
    vector = _sample_vector()
    payload = _payload(
        "single_user",
        data="hello aerospike",
        category="test",
    )
    db.insert(vectors=[vector], payloads=[payload], ids=["single-id"])

    result = db.get("single-id")
    assert result is not None
    assert result.id == "single-id"
    assert result.payload == payload


def test_insert_multiple_records(db):
    """3.3 RED: batch insert with multiple records round-trips each one."""
    ids = ["batch-a", "batch-b", "batch-c"]
    vectors = [_sample_vector(i) for i in range(len(ids))]
    payloads = [_payload(f"batch_user_{i}", data=f"memory {i}") for i in range(len(ids))]

    db.insert(vectors=vectors, payloads=payloads, ids=ids)

    for idx, memory_id in enumerate(ids):
        result = db.get(memory_id)
        assert result is not None
        assert result.id == memory_id
        assert result.payload == payloads[idx]


def test_insert_upserts_existing_id(db):
    """3.5 RED: inserting the same id twice replaces the record rather than raising."""
    vector = _sample_vector()
    original_payload = _payload("upsert_user", data="original")
    new_payload = _payload("upsert_user", data="replaced")

    db.insert(vectors=[vector], payloads=[original_payload], ids=["upsert-id"])
    db.insert(vectors=[vector], payloads=[new_payload], ids=["upsert-id"])

    result = db.get("upsert-id")
    assert result is not None
    assert result.payload == new_payload


def test_delete_removes_record(db):
    """3.7 RED: delete(id) followed by get(id) returns None."""
    db.insert(
        vectors=[_sample_vector()],
        payloads=[_payload("delete_user")],
        ids=["delete-id"],
    )
    db.delete("delete-id")
    assert db.get("delete-id") is None


def test_long_metadata_key_roundtrips(db):
    """3.9 RED: metadata keys longer than 15 chars survive insert/get."""
    payload = _payload(
        "long_key_user",
        **{"a_very_long_custom_field": "stored value"},
    )
    db.insert(vectors=[_sample_vector()], payloads=[payload], ids=["long-key-id"])

    result = db.get("long-key-id")
    assert result.payload == payload
    assert result.payload["a_very_long_custom_field"] == "stored value"


def test_insert_rejects_missing_ids(db):
    """3.10 RED: insert() raises a clear error when ids is None."""
    with pytest.raises(ValueError, match="ids is required"):
        db.insert(vectors=[_sample_vector()], payloads=[_payload("missing_ids_user")], ids=None)


def test_insert_rejects_mismatched_ids_length(db):
    """3.11 RED: insert() raises a clear error when ids is shorter than vectors."""
    with pytest.raises(ValueError, match="ids length .* must match vectors length"):
        db.insert(
            vectors=[_sample_vector(), _sample_vector()],
            payloads=[_payload("mismatch_user"), _payload("mismatch_user")],
            ids=["only-one-id"],
        )


def test_insert_rejects_mismatched_payloads_length(db):
    """3.12 RED: insert() raises a clear error when payloads is shorter than vectors."""
    with pytest.raises(ValueError, match="payloads length .* must match vectors length"):
        db.insert(
            vectors=[_sample_vector(), _sample_vector()],
            payloads=[_payload("mismatch_user")],
            ids=["id-1", "id-2"],
        )


# 4. Update semantics


def test_update_replaces_full_payload(db):
    """4.1 RED: update replaces the full payload, including metadata map."""
    original = _payload("update_user", data="before", old_meta="keep me")
    db.insert(vectors=[_sample_vector()], payloads=[original], ids=["update-full-id"])

    new_vector = _sample_vector(2.0)
    new_payload = _payload("update_user", data="after")
    db.update("update-full-id", vector=new_vector, payload=new_payload)

    result = db.get("update-full-id")
    assert result.payload == new_payload
    assert "old_meta" not in result.payload


def test_update_without_vector_preserves_embedding(db):
    """4.3 RED: update(vector=None) keeps the original vector searchable."""
    original_vector = _sample_vector(1.0)
    payload = _payload("preserve_user")
    db.insert(vectors=[original_vector], payloads=[payload], ids=["preserve-id"])

    db.update("preserve-id", vector=None, payload=_payload("preserve_user", data="updated"))

    results = db.search(
        "query",
        vectors=original_vector,
        filters={"user_id": "preserve_user"},
        top_k=1,
    )
    assert len(results) == 1
    assert results[0].id == "preserve-id"


# 5. Size and record limits


def test_insert_oversized_record_raises(db):
    """5.1 RED: insert rejects a record larger than max_record_bytes before writing."""
    payload = _payload("size_user", data="x" * 2_000_000)
    with pytest.raises(Exception, match="record size|max_record_bytes|too large"):
        db.insert(vectors=[_sample_vector()], payloads=[payload], ids=["oversized-insert-id"])


def test_update_oversized_record_raises(db):
    """5.3 RED: update rejects a record larger than max_record_bytes before writing."""
    db.insert(vectors=[_sample_vector()], payloads=[_payload("size_user_update")], ids=["oversized-update-id"])
    oversized_payload = _payload("size_user_update", data="x" * 2_000_000)
    with pytest.raises(Exception, match="record size|max_record_bytes|too large"):
        db.update("oversized-update-id", vector=_sample_vector(), payload=oversized_payload)


# 6. Filtered list


def test_list_filters_by_user_id(db):
    """6.1: list(filters={"user_id": ...}) returns matching records.

    Ordering is intentionally unspecified (mem0 does not require it); the
    provider uses a short-circuiting limit() rather than a full-set sort.
    """
    user_id = "list_user_1"
    t1 = _iso(datetime(2025, 6, 1, 10, 0, 0))
    t2 = _iso(datetime(2025, 6, 1, 11, 0, 0))
    t3 = _iso(datetime(2025, 6, 1, 12, 0, 0))

    db.insert(
        vectors=[_sample_vector(i) for i in range(3)],
        payloads=[
            _payload(user_id, created_at=t1),
            _payload(user_id, created_at=t2),
            _payload("other_user", created_at=t3),
        ],
        ids=["lu-1", "lu-2", "lu-other"],
    )

    results = db.list(filters={"user_id": user_id}, top_k=10)
    memories = results[0]
    assert len(memories) == 2
    assert all(m.payload["user_id"] == user_id for m in memories)
    assert {m.id for m in memories} == {"lu-1", "lu-2"}


def test_list_filters_by_user_id_eq_operator(db):
    """6.2: list(filters={"user_id": {"eq": ...}}) is treated as scoped."""
    user_id = "list_user_eq_operator"
    db.insert(
        vectors=[_sample_vector(i) for i in range(3)],
        payloads=[
            _payload(user_id, created_at=_iso(datetime(2025, 6, 1, 10, 0, 0))),
            _payload(user_id, created_at=_iso(datetime(2025, 6, 1, 11, 0, 0))),
            _payload("other_user", created_at=_iso(datetime(2025, 6, 1, 12, 0, 0))),
        ],
        ids=["lueq-1", "lueq-2", "lueq-other"],
    )

    results = db.list(filters={"user_id": {"eq": user_id}}, top_k=10)
    memories = results[0]
    assert len(memories) == 2
    assert all(m.payload["user_id"] == user_id for m in memories)
    assert {m.id for m in memories} == {"lueq-1", "lueq-2"}


def test_list_filters_by_agent_id_only(db):
    """6.3 RED: list(filters={"agent_id": ...}) works without user_id or run_id."""
    agent_id = "agent-only"
    db.insert(
        vectors=[_sample_vector(i) for i in range(2)],
        payloads=[
            _payload("u1", agent_id=agent_id),
            _payload("u2", agent_id="other-agent"),
        ],
        ids=["agent-match", "agent-miss"],
    )

    results = db.list(filters={"agent_id": agent_id}, top_k=10)
    memories = results[0]
    assert len(memories) == 1
    assert memories[0].payload["agent_id"] == agent_id
    assert memories[0].id == "agent-match"


def test_list_filters_by_run_id_only(db):
    """6.4 RED: list(filters={"run_id": ...}) works without user_id or agent_id."""
    run_id = "run-only"
    db.insert(
        vectors=[_sample_vector(i) for i in range(2)],
        payloads=[
            _payload("u1", run_id=run_id),
            _payload("u2", run_id="other-run"),
        ],
        ids=["run-match", "run-miss"],
    )

    results = db.list(filters={"run_id": run_id}, top_k=10)
    memories = results[0]
    assert len(memories) == 1
    assert memories[0].payload["run_id"] == run_id
    assert memories[0].id == "run-match"


def test_list_unscoped_filter_rejects_scan_by_default(db):
    """6.6 RED: a metadata-only filter raises when allow_scans_with_where is False."""
    with pytest.raises(Exception, match="scan|scoping|user_id|agent_id|run_id"):
        db.list(filters={"category": "work"}, top_k=10)


def test_list_unscoped_filter_allows_scan_when_configured(db_allowing_scans):
    """6.7 RED: the same metadata-only filter succeeds when scans are explicitly allowed."""
    db = db_allowing_scans
    db.insert(
        vectors=[_sample_vector()],
        payloads=[_payload("scan_user", category="work")],
        ids=["scan-id"],
    )
    results = db.list(filters={"category": "work"}, top_k=10)
    memories = results[0]
    assert len(memories) == 1
    assert memories[0].id == "scan-id"


@pytest.mark.parametrize(
    "operator, value, expected_ids",
    [
        ("eq", "alice", {"meta-eq", "meta-gte"}),
        ("ne", "alice", {"meta-ne", "meta-gt", "meta-lt", "meta-contains", "meta-icontains"}),
        ("gt", 10, {"meta-ne", "meta-gt", "meta-gte", "meta-lt"}),
        ("gte", 20, {"meta-ne", "meta-gt", "meta-gte"}),
        ("lt", 20, {"meta-eq", "meta-lt"}),
        ("lte", 20, {"meta-eq", "meta-ne", "meta-gte", "meta-lt"}),
        ("in", ["alice", "bob"], {"meta-eq", "meta-gte", "meta-ne"}),
        ("nin", ["alice"], {"meta-ne", "meta-gt", "meta-lt", "meta-contains", "meta-icontains"}),
        ("contains", "work", {"meta-contains"}),
        ("icontains", "WORK", {"meta-contains", "meta-icontains"}),
    ],
)
def test_list_metadata_operators(db, operator, value, expected_ids):
    """6.9 RED: list supports metadata comparison operators."""
    user_id = "meta_ops_user"
    db.insert(
        vectors=[_sample_vector(i) for i in range(7)],
        payloads=[
            _payload(user_id, name="alice", score=10),
            _payload(user_id, name="bob", score=20),
            _payload(user_id, name="charlie", score=30),
            _payload(user_id, name="alice", score=20),
            _payload(user_id, name="dave", score=15),
            _payload(user_id, name="workitem", category="work docs"),
            _payload(user_id, name="WorkItem", category="WORK docs"),
        ],
        ids=[
            "meta-eq",
            "meta-ne",
            "meta-gt",
            "meta-gte",
            "meta-lt",
            "meta-contains",
            "meta-icontains",
        ],
    )

    if operator in ("contains", "icontains"):
        field = "category"
    elif operator in ("eq", "ne", "in", "nin"):
        field = "name"
    else:
        field = "score"

    filters = {"user_id": user_id, field: {operator: value}}
    results = db.list(filters=filters, top_k=20)
    memories = results[0]
    assert {m.id for m in memories} == expected_ids


def test_list_logical_combinators(db):
    """6.11 RED: list handles $or and $not combinators alongside a required user_id."""
    user_id = "logic_user"
    db.insert(
        vectors=[_sample_vector(i) for i in range(3)],
        payloads=[
            _payload(user_id, name="alice", color="red"),
            _payload(user_id, name="bob", color="blue"),
            _payload(user_id, name="charlie", color="green"),
        ],
        ids=["logic-1", "logic-2", "logic-3"],
    )

    results = db.list(
        filters={
            "user_id": user_id,
            "$or": [
                {"name": {"eq": "alice"}},
                {"name": {"eq": "bob"}},
            ],
        },
        top_k=10,
    )
    memories = results[0]
    assert {m.id for m in memories} == {"logic-1", "logic-2"}

    results = db.list(
        filters={
            "user_id": user_id,
            "$not": [
                {"name": {"eq": "alice"}},
            ],
        },
        top_k=10,
    )
    memories = results[0]
    assert {m.id for m in memories} == {"logic-2", "logic-3"}


def test_list_logical_combinators_multi_field_and(db):
    """list AND's multiple fields together within a single $or/$not branch."""
    user_id = "logic_and_user"
    db.insert(
        vectors=[_sample_vector(i) for i in range(3)],
        payloads=[
            _payload(user_id, name="alice", color="red"),
            _payload(user_id, name="alice", color="blue"),
            _payload(user_id, name="bob", color="red"),
        ],
        ids=["logicand-1", "logicand-2", "logicand-3"],
    )

    results = db.list(
        filters={
            "user_id": user_id,
            "$or": [
                {"name": {"eq": "alice"}, "color": {"eq": "red"}},
                {"name": {"eq": "bob"}, "color": {"eq": "red"}},
            ],
        },
        top_k=10,
    )
    memories = results[0]
    assert {m.id for m in memories} == {"logicand-1", "logicand-3"}


def test_list_logical_combinators_nested(db):
    """$or/$not branches may themselves contain nested $or/$not sub-conditions."""
    user_id = "logic_nested_user"
    db.insert(
        vectors=[_sample_vector(i) for i in range(4)],
        payloads=[
            _payload(user_id, name="alice", color="red"),
            _payload(user_id, name="bob", color="blue"),
            _payload(user_id, name="charlie", color="green"),
            _payload(user_id, name="dave", color="red"),
        ],
        ids=["nested-1", "nested-2", "nested-3", "nested-4"],
    )

    # Nested $or inside a $or branch: name == "charlie" OR (name == "alice" OR name == "dave")
    results = db.list(
        filters={
            "user_id": user_id,
            "$or": [
                {"name": {"eq": "charlie"}},
                {"$or": [{"name": {"eq": "alice"}}, {"name": {"eq": "dave"}}]},
            ],
        },
        top_k=10,
    )
    memories = results[0]
    assert {m.id for m in memories} == {"nested-1", "nested-3", "nested-4"}

    # Nested $not inside a $not branch: NOT (color == "red" AND NOT (name == "dave"))
    # excludes only the red record that isn't dave, i.e. excludes "alice"/red.
    results = db.list(
        filters={
            "user_id": user_id,
            "$not": [
                {"color": {"eq": "red"}, "$not": [{"name": {"eq": "dave"}}]},
            ],
        },
        top_k=10,
    )
    memories = results[0]
    assert {m.id for m in memories} == {"nested-2", "nested-3", "nested-4"}


def test_list_filter_values_are_literal(db):
    """6.13 RED: filter values containing AEL-meaningful characters match only the literal value."""
    user_id = "literal_user"
    db.insert(
        vectors=[_sample_vector(i) for i in range(2)],
        payloads=[
            _payload(user_id, name="O'Brien"),
            _payload(user_id, name='"; drop'),
        ],
        ids=["literal-1", "literal-2"],
    )

    for memory_id, name in [("literal-1", "O'Brien"), ("literal-2", '"; drop')]:
        results = db.list(filters={"user_id": user_id, "name": {"eq": name}}, top_k=10)
        memories = results[0]
        assert len(memories) == 1
        assert memories[0].id == memory_id


def test_list_empty_in_nin_are_constant(db):
    """in/nin against an empty list are constants, independent of the field's type.

    Regression test: an empty `in`/`nin` list leaves no sample value to infer the
    field's Exp type from, so the previous implementation defaulted the sample to "".
    For a non-string metadata field (e.g. a numeric `score`), that type-mismatched
    map read evaluates to "unknown" rather than false, and NOT() does not flip
    "unknown" to true - so `nin: []` incorrectly matched nothing instead of everything.
    """
    user_id = "empty_in_nin_user"
    db.insert(
        vectors=[_sample_vector(i) for i in range(2)],
        payloads=[
            _payload(user_id, score=10),
            _payload(user_id, score=20),
        ],
        ids=["emptyinnin-1", "emptyinnin-2"],
    )

    results = db.list(filters={"user_id": user_id, "score": {"in": []}}, top_k=10)
    assert {m.id for m in results[0]} == set()

    results = db.list(filters={"user_id": user_id, "score": {"nin": []}}, top_k=10)
    assert {m.id for m in results[0]} == {"emptyinnin-1", "emptyinnin-2"}


def test_list_unsupported_operator_raises(db):
    """6.15 RED: an unsupported operator raises a clear error instead of being ignored."""
    with pytest.raises(Exception, match="Unsupported|unsupported|operator"):
        db.list(filters={"user_id": "u", "name": {"$regex": ".*"}}, top_k=10)


# 7. search()


def test_search_ranks_by_cosine_similarity(db):
    """7.1 RED: search returns top-k results ordered by descending similarity."""
    user_id = "search_user_1"
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.9, 0.1, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0, 0.0]

    db.insert(
        vectors=[v1, v2, v3],
        payloads=[
            _payload(user_id, data="first"),
            _payload(user_id, data="second"),
            _payload(user_id, data="third"),
        ],
        ids=["rank-1", "rank-2", "rank-3"],
    )

    results = db.search("query", vectors=v1, filters={"user_id": user_id}, top_k=2)
    assert len(results) <= 2
    assert all(r.payload["user_id"] == user_id for r in results)


def test_search_filters_by_user_id_eq_operator(db):
    """7.2: search(filters={"user_id": {"eq": ...}}) is treated as scoped."""
    user_id = "search_user_eq_operator"
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.9, 0.1, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0, 0.0]

    db.insert(
        vectors=[v1, v2, v3],
        payloads=[
            _payload(user_id, data="first"),
            _payload(user_id, data="second"),
            _payload("other_user", data="third"),
        ],
        ids=["rankeq-1", "rankeq-2", "rankeq-3"],
    )

    results = db.search("query", vectors=v1, filters={"user_id": {"eq": user_id}}, top_k=2)
    assert len(results) <= 2
    assert all(r.payload["user_id"] == user_id for r in results)
    assert "rankeq-3" not in {r.id for r in results}
    assert results[0].score >= results[1].score
    assert {r.id for r in results} <= {"rankeq-1", "rankeq-2"}


def test_search_scoped_by_agent_id_and_run_id(db):
    """7.3 RED: search works with agent_id-only and run_id-only filters."""
    agent_id = "search_agent"
    run_id = "search_run"
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.9, 0.1, 0.0, 0.0]

    db.insert(
        vectors=[v1, v2],
        payloads=[
            _payload("u1", agent_id=agent_id, data="agent match"),
            _payload("u2", run_id=run_id, data="run match"),
        ],
        ids=["search-agent", "search-run"],
    )

    agent_results = db.search("q", vectors=v1, filters={"agent_id": agent_id}, top_k=10)
    assert len(agent_results) == 1
    assert agent_results[0].id == "search-agent"

    run_results = db.search("q", vectors=v1, filters={"run_id": run_id}, top_k=10)
    assert len(run_results) == 1
    assert run_results[0].id == "search-run"


def test_search_unscoped_filter_rejects_scan_by_default(db):
    """7.5 RED: search without a scoping key raises unless scans are allowed."""
    with pytest.raises(Exception, match="scan|scoping|user_id|agent_id|run_id"):
        db.search("query", vectors=_sample_vector(), filters={"category": "x"}, top_k=10)


def test_keyword_search_returns_none(db):
    """7.7 RED: keyword_search returns None."""
    assert db.keyword_search("anything", top_k=5, filters={"user_id": "u"}) is None


# 7b. search() filter coverage — the same _compile_filters matrix as list(),
# exercised through the score + order_by + top_k path.


def _seed_search_ops_corpus(db, user_id: str = "search_ops_user") -> str:
    """Seed the same labeled corpus used by the list operator matrix."""
    db.insert(
        vectors=[_sample_vector(i + 1) for i in range(7)],
        payloads=[
            _payload(user_id, name="alice", score=10),
            _payload(user_id, name="bob", score=20),
            _payload(user_id, name="charlie", score=30),
            _payload(user_id, name="alice", score=20),
            _payload(user_id, name="dave", score=15),
            _payload(user_id, name="workitem", category="work docs"),
            _payload(user_id, name="WorkItem", category="WORK docs"),
        ],
        ids=[
            "sop-eq",
            "sop-ne",
            "sop-gt",
            "sop-gte",
            "sop-lt",
            "sop-contains",
            "sop-icontains",
        ],
    )
    return user_id


def _search_ids(db, filters, top_k=20, vector=None):
    return [r.id for r in db.search("q", vectors=vector or _sample_vector(), filters=filters, top_k=top_k)]


@pytest.mark.parametrize(
    "operator, value, expected_ids",
    [
        ("eq", "alice", {"sop-eq", "sop-gte"}),
        ("ne", "alice", {"sop-ne", "sop-gt", "sop-lt", "sop-contains", "sop-icontains"}),
        ("gt", 10, {"sop-ne", "sop-gt", "sop-gte", "sop-lt"}),
        ("gte", 20, {"sop-ne", "sop-gt", "sop-gte"}),
        ("lt", 20, {"sop-eq", "sop-lt"}),
        ("lte", 20, {"sop-eq", "sop-ne", "sop-gte", "sop-lt"}),
        ("in", ["alice", "bob"], {"sop-eq", "sop-gte", "sop-ne"}),
        ("nin", ["alice"], {"sop-ne", "sop-gt", "sop-lt", "sop-contains", "sop-icontains"}),
        ("contains", "work", {"sop-contains"}),
        ("icontains", "WORK", {"sop-contains", "sop-icontains"}),
    ],
)
def test_search_metadata_operators(db, operator, value, expected_ids):
    """Metadata comparison operators filter the candidate set before ranking."""
    user_id = _seed_search_ops_corpus(db)

    if operator in ("contains", "icontains"):
        field = "category"
    elif operator in ("eq", "ne", "in", "nin"):
        field = "name"
    else:
        field = "score"

    ids = _search_ids(db, {"user_id": user_id, field: {operator: value}})
    assert set(ids) == expected_ids


def test_search_logical_combinators(db):
    """$or and $not combine with the required user_id scope in search()."""
    user_id = "search_logic_user"
    db.insert(
        vectors=[_sample_vector(i + 1) for i in range(3)],
        payloads=[
            _payload(user_id, name="alice", color="red"),
            _payload(user_id, name="bob", color="blue"),
            _payload(user_id, name="charlie", color="green"),
        ],
        ids=["slogic-1", "slogic-2", "slogic-3"],
    )

    ids = _search_ids(
        db,
        {
            "user_id": user_id,
            "$or": [
                {"name": {"eq": "alice"}},
                {"name": {"eq": "bob"}},
            ],
        },
    )
    assert set(ids) == {"slogic-1", "slogic-2"}

    ids = _search_ids(
        db,
        {
            "user_id": user_id,
            "$not": [
                {"name": {"eq": "alice"}},
            ],
        },
    )
    assert set(ids) == {"slogic-2", "slogic-3"}


def test_search_logical_combinators_multi_field_and(db):
    """Multiple fields within one $or branch AND together in search()."""
    user_id = "search_logic_and_user"
    db.insert(
        vectors=[_sample_vector(i + 1) for i in range(3)],
        payloads=[
            _payload(user_id, name="alice", color="red"),
            _payload(user_id, name="alice", color="blue"),
            _payload(user_id, name="bob", color="red"),
        ],
        ids=["sland-1", "sland-2", "sland-3"],
    )

    ids = _search_ids(
        db,
        {
            "user_id": user_id,
            "$or": [
                {"name": {"eq": "alice"}, "color": {"eq": "red"}},
                {"name": {"eq": "bob"}, "color": {"eq": "red"}},
            ],
        },
    )
    assert set(ids) == {"sland-1", "sland-3"}


def test_search_logical_combinators_nested(db):
    """Nested $or/$not branches evaluate correctly through the search path."""
    user_id = "search_logic_nested_user"
    db.insert(
        vectors=[_sample_vector(i + 1) for i in range(4)],
        payloads=[
            _payload(user_id, name="alice", color="red"),
            _payload(user_id, name="bob", color="blue"),
            _payload(user_id, name="charlie", color="green"),
            _payload(user_id, name="dave", color="red"),
        ],
        ids=["snested-1", "snested-2", "snested-3", "snested-4"],
    )

    ids = _search_ids(
        db,
        {
            "user_id": user_id,
            "$or": [
                {"name": {"eq": "charlie"}},
                {"$or": [{"name": {"eq": "alice"}}, {"name": {"eq": "dave"}}]},
            ],
        },
    )
    assert set(ids) == {"snested-1", "snested-3", "snested-4"}

    ids = _search_ids(
        db,
        {
            "user_id": user_id,
            "$not": [
                {"color": {"eq": "red"}, "$not": [{"name": {"eq": "dave"}}]},
            ],
        },
    )
    assert set(ids) == {"snested-2", "snested-3", "snested-4"}


def test_search_metadata_only_scan_when_configured(db_allowing_scans):
    """A metadata-only filter succeeds on the opt-in scan path."""
    db = db_allowing_scans
    db.insert(
        vectors=[_sample_vector()],
        payloads=[_payload("scan_user", category="work")],
        ids=["search-scan-id"],
    )
    ids = _search_ids(db, {"category": "work"})
    assert ids == ["search-scan-id"]


def test_search_filter_values_are_literal(db):
    """Quote/AEL-meaningful filter values match only their literal value."""
    user_id = "search_literal_user"
    db.insert(
        vectors=[_sample_vector(i + 1) for i in range(2)],
        payloads=[
            _payload(user_id, name="O'Brien"),
            _payload(user_id, name='"; drop'),
        ],
        ids=["sliteral-1", "sliteral-2"],
    )

    for memory_id, name in [("sliteral-1", "O'Brien"), ("sliteral-2", '"; drop')]:
        ids = _search_ids(db, {"user_id": user_id, "name": {"eq": name}})
        assert ids == [memory_id]


# 7c. top_k and candidate-set edge cases


def test_search_top_k_exceeds_candidate_count(db):
    """top_k larger than the filtered set returns just the candidates."""
    user_id = "search_topk_overflow_user"
    db.insert(
        vectors=[_sample_vector(i + 1) for i in range(3)],
        payloads=[_payload(user_id, data=f"m{i}") for i in range(3)],
        ids=["tko-1", "tko-2", "tko-3"],
    )
    results = db.search("q", vectors=_sample_vector(), filters={"user_id": user_id}, top_k=10)
    assert len(results) == 3
    assert {r.id for r in results} == {"tko-1", "tko-2", "tko-3"}


def test_search_top_k_one(db):
    """top_k=1 returns only the single most-similar candidate."""
    user_id = "search_topk_one_user"
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.9, 0.1, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0, 0.0]
    db.insert(
        vectors=[v1, v2, v3],
        payloads=[_payload(user_id) for _ in range(3)],
        ids=["tk1-best", "tk1-second", "tk1-far"],
    )
    results = db.search("q", vectors=v1, filters={"user_id": user_id}, top_k=1)
    assert [r.id for r in results] == ["tk1-best"]


def test_search_empty_candidate_set(db):
    """A scope matching zero records returns [] rather than raising."""
    results = db.search(
        "q",
        vectors=_sample_vector(),
        filters={"user_id": "definitely-no-such-user"},
        top_k=10,
    )
    assert results == []


def test_search_score_ordering_with_near_ties(db):
    """Near-identical vectors still produce a deterministic descending order."""
    user_id = "search_ties_user"
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [1.0, 1e-9, 0.0, 0.0]
    v3 = [0.99, 0.01, 0.0, 0.0]
    db.insert(
        vectors=[v2, v3, v1],
        payloads=[_payload(user_id) for _ in range(3)],
        ids=["tie-a", "tie-b", "tie-c"],
    )
    results = db.search("q", vectors=v1, filters={"user_id": user_id}, top_k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # Cosine similarity of v1-vs-v1 (1.0) edges out v2 (~1 - 1e-18) and v3 (~0.9999).
    assert results[0].id == "tie-c"


# 7d. Result shape


def test_search_payload_matches_get(db):
    """search() reconstructs the same flat payload get() returns."""
    user_id = "search_shape_user"
    payload = _payload(
        user_id,
        data="shape check",
        category="docs",
        extra_field="x",
    )
    db.insert(vectors=[_sample_vector()], payloads=[payload], ids=["shape-1"])

    got = db.get("shape-1")
    found = db.search("q", vectors=_sample_vector(), filters={"user_id": user_id}, top_k=1)
    assert len(found) == 1
    assert found[0].payload == got.payload == payload


def test_search_scores_in_range_and_ordered(db):
    """Scores are cosine-range similarities in non-increasing order."""
    user_id = "search_score_range_user"
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    db.insert(
        vectors=vectors,
        payloads=[_payload(user_id) for _ in vectors],
        ids=["sr-1", "sr-2", "sr-3"],
    )
    results = db.search("q", vectors=vectors[0], filters={"user_id": user_id}, top_k=10)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert all(s is not None and -1.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_search_result_ids_are_memory_ids(db):
    """Top-K results surface the stored memory_id via sendKey, not digests."""
    user_id = "search_id_user"
    db.insert(
        vectors=[_sample_vector(i + 1) for i in range(2)],
        payloads=[_payload(user_id) for _ in range(2)],
        ids=["mid-abc-123", "mid-def-456"],
    )
    ids = _search_ids(db, {"user_id": user_id})
    assert set(ids) == {"mid-abc-123", "mid-def-456"}


# 7e. Scope-field coverage


def test_search_scoped_by_run_id_only(db):
    """run_id alone is sufficient to scope a search."""
    run_id = "search_run_only"
    db.insert(
        vectors=[_sample_vector(i + 1) for i in range(2)],
        payloads=[
            _payload("u1", run_id=run_id),
            _payload("u2", run_id="other-run"),
        ],
        ids=["srun-match", "srun-miss"],
    )
    ids = _search_ids(db, {"run_id": run_id})
    assert ids == ["srun-match"]


def test_search_scoped_by_agent_id_only(db):
    """agent_id alone is sufficient to scope a search."""
    agent_id = "search_agent_only"
    db.insert(
        vectors=[_sample_vector(i + 1) for i in range(2)],
        payloads=[
            _payload("u1", agent_id=agent_id),
            _payload("u2", agent_id="other-agent"),
        ],
        ids=["sagent-match", "sagent-miss"],
    )
    ids = _search_ids(db, {"agent_id": agent_id})
    assert ids == ["sagent-match"]


def test_search_multi_scope_filter(db):
    """user_id + agent_id intersect regardless of which is the index predicate."""
    user_id = "search_multiscope_user"
    agent_id = "search_multiscope_agent"
    db.insert(
        vectors=[_sample_vector(i + 1) for i in range(3)],
        payloads=[
            _payload(user_id, agent_id=agent_id),
            _payload(user_id, agent_id="other-agent"),
            _payload("other-user", agent_id=agent_id),
        ],
        ids=["ms-both", "ms-user-only", "ms-agent-only"],
    )
    ids = _search_ids(db, {"user_id": user_id, "agent_id": agent_id})
    assert ids == ["ms-both"]
