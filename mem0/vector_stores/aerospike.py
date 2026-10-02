import json
from datetime import datetime, timezone

try:
    from aerospike_async import BatchWritePolicy, ExpOperation, Operation
    from aerospike_sdk import (
        Behavior,
        DataSet,
        Exp,
        ExpType,
        Filter,
        MapReturnType,
        Order,
        OrderByType,
        ResultCode,
        StringWriteFlags,
        Vector,
    )
    from aerospike_sdk.exceptions import AerospikeError
    from aerospike_sdk.policy.behavior_settings import Mode, OpKind, OpShape
    from aerospike_sdk.policy.policy_mapper import to_batch_policy
    from aerospike_sdk.sync import ClusterDefinition
except ImportError as e:
    raise ImportError(
        "Aerospike vector store support requires the preview 'aerospike-sdk' package. "
        "Install it with the vector-stores extra: pip install 'mem0ai[vector-stores]', "
        "or directly: pip install 'aerospike-sdk'."
    ) from e

from mem0.vector_stores.base import VectorStoreBase

# Fixed core-bin map: these payload keys get dedicated bins. Every other
# payload key is bundled into the single `metadata` Map (CDT) bin, since
# Aerospike bin names are capped at 15 characters server-side.
CORE_STRING_FIELDS = ("data", "hash", "user_id", "agent_id", "run_id", "text_lemmatized", "attributed_to")
TIMESTAMP_FIELDS = ("created_at", "updated_at")
SCOPE_FIELDS = ("user_id", "agent_id", "run_id")
SUPPORTED_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"}

EMBEDDING_BIN = "embedding"
METADATA_BIN = "metadata"


class MemoryResult:
    def __init__(self, id: str, payload: dict, score: float | None = None):
        self.id = id
        self.payload = payload
        self.score = score


def _iso_to_epoch_ms(value: str) -> int:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _epoch_ms_to_iso(ms: int) -> str:
    seconds, millis = divmod(int(ms), 1000)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=millis * 1000)
    return dt.isoformat(timespec="milliseconds")


def _exp_type_for(value):
    if isinstance(value, bool):
        return ExpType.BOOL
    if isinstance(value, int):
        return ExpType.INT
    if isinstance(value, float):
        return ExpType.FLOAT
    if isinstance(value, str):
        return ExpType.STRING
    raise ValueError(f"Unsupported filter value type: {type(value).__name__}")


def _field_expr(key: str, sample_value):
    """Build the Exp for a field: a core bin, a timestamp bin, or a metadata map key."""
    if key in CORE_STRING_FIELDS:
        return Exp.string_bin(key)
    if key in TIMESTAMP_FIELDS:
        return Exp.int_bin(f"{key}_ms")
    value_type = _exp_type_for(sample_value)
    return Exp.map_get_by_key(MapReturnType.VALUE, value_type, Exp.string_val(key), Exp.map_bin(METADATA_BIN), [])


def _compile_operator(key: str, op: str, value):
    if op not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"Unsupported filter operator '{op}' for field '{key}'. Supported operators: {sorted(SUPPORTED_OPERATORS)}"
        )
    if op == "eq":
        return Exp.eq(_field_expr(key, value), Exp.val(value))
    if op == "ne":
        return Exp.ne(_field_expr(key, value), Exp.val(value))
    if op == "gt":
        return Exp.gt(_field_expr(key, value), Exp.val(value))
    if op == "gte":
        return Exp.ge(_field_expr(key, value), Exp.val(value))
    if op == "lt":
        return Exp.lt(_field_expr(key, value), Exp.val(value))
    if op == "lte":
        return Exp.le(_field_expr(key, value), Exp.val(value))
    if op in ("in", "nin"):
        values = list(value)
        if not values:
            return Exp.val(op == "nin")
        sample = values[0]
        expr = Exp.in_list(_field_expr(key, sample), Exp.val(values))
        return Exp.not_(expr) if op == "nin" else expr
    if op == "contains":
        return Exp.string_contains(Exp.string_val(str(value)), _field_expr(key, value))
    # icontains
    lowered_field = Exp.string_lower(int(StringWriteFlags.DEFAULT), _field_expr(key, value))
    return Exp.string_contains(Exp.string_val(str(value).lower()), lowered_field)


def _compile_field(key: str, condition):
    """Build the Exp for one field: literal equality, or an operator dict."""
    if not isinstance(condition, dict):
        return Exp.eq(_field_expr(key, condition), Exp.val(condition))
    exprs = [_compile_operator(key, op, value) for op, value in condition.items()]
    return exprs[0] if len(exprs) == 1 else Exp.and_(exprs)


def _compile_filters(filters, exclude_key=None):
    """Compile a mem0 filter dict into an Exp tree. Never interpolates values into AEL strings.

    Recurses into itself for $or/$not sub-conditions, so a sub-condition may
    itself contain nested $or/$not/field clauses.
    """
    if not filters:
        return None
    clauses = []
    for key, value in filters.items():
        if key == exclude_key:
            continue
        if key == "$or":
            sub = [c for c in (_compile_filters(cond) for cond in value) if c is not None]
            if sub:
                clauses.append(sub[0] if len(sub) == 1 else Exp.or_(sub))
        elif key == "$not":
            for cond in value:
                compiled = _compile_filters(cond)
                if compiled is not None:
                    clauses.append(Exp.not_(compiled))
        else:
            clauses.append(_compile_field(key, value))
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else Exp.and_(clauses)


class AerospikeDB(VectorStoreBase):
    """Aerospike-backed vector store provider."""

    def __init__(
        self,
        namespace: str,
        collection_name: str,
        embedding_model_dims: int,
        host: str = "localhost",
        port: int = 3000,
        allow_scans_with_where: bool = False,
        max_record_bytes: int = 943718,
    ):
        self.namespace = namespace
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.host = host
        self.port = port
        self.allow_scans_with_where = allow_scans_with_where
        self.max_record_bytes = max_record_bytes
        self.dataset = DataSet.of(namespace, collection_name)

        self._cluster = ClusterDefinition(host, port).connect()
        # sendKey = true so memory_id can be recovered from secondary-index and
        # Top-K query results, which otherwise return only digests.
        behavior = Behavior.DEFAULT.derive_with_changes(f"mem0-aerospike-{collection_name}", send_key=True)
        self.session = self._cluster.create_session(behavior)

        batch_write_settings = behavior.get_settings(OpKind.WRITE_RETRYABLE, OpShape.BATCH, Mode.AP)
        self._batch_policy = to_batch_policy(batch_write_settings)
        self._batch_write_policy = BatchWritePolicy()
        self._batch_write_policy.send_key = batch_write_settings.send_key
        self._batch_write_policy.commit_level = batch_write_settings.commit_level
        self._batch_write_policy.durable_delete = batch_write_settings.durable_delete

        self.create_col()

    # -- Collection lifecycle --------------------------------------------------

    def create_col(self, name=None, vector_size=None, distance=None):
        collection_name = name or self.collection_name
        dataset = DataSet.of(self.namespace, collection_name) if name else self.dataset

        existing = {
            idx.get("name")
            for idx in self.session.list_indexes()
            if idx.get("namespace") == self.namespace and idx.get("set") == collection_name
        }
        for field in SCOPE_FIELDS:
            index_name = f"{collection_name}_{field}_idx"
            if index_name in existing:
                continue
            task = self.session.index(dataset=dataset).on_bin(field).named(index_name).string().create()
            task.wait_till_complete_blocking()

    def list_cols(self):
        return [s.name for s in self.session.info().sets(self.namespace)]

    def delete_col(self):
        task = self.session.truncate(self.dataset)
        if task is not None:
            task.wait_till_complete_blocking()

    def reset(self):
        self.delete_col()
        self.create_col()

    def col_info(self):
        for set_detail in self.session.info().sets(self.namespace):
            if set_detail.name == self.collection_name:
                return {"name": self.collection_name, "count": set_detail.objects}
        return {"name": self.collection_name, "count": 0}

    # -- Payload <-> bins translation ------------------------------------------

    def _payload_to_bins(self, payload: dict, vector=None) -> dict:
        bins = {}
        metadata = {}
        for key, value in (payload or {}).items():
            if key in TIMESTAMP_FIELDS:
                bins[f"{key}_ms"] = _iso_to_epoch_ms(value)
            elif key in CORE_STRING_FIELDS:
                bins[key] = value
            else:
                metadata[key] = value
        bins[METADATA_BIN] = metadata
        if vector is not None:
            bins[EMBEDDING_BIN] = Vector(vector)
        return bins

    def _bins_to_payload(self, bins: dict) -> dict:
        payload = {}
        for key in CORE_STRING_FIELDS:
            if bins.get(key) is not None:
                payload[key] = bins[key]
        for key in TIMESTAMP_FIELDS:
            ms_value = bins.get(f"{key}_ms")
            if ms_value is not None:
                payload[key] = _epoch_ms_to_iso(ms_value)
        metadata = bins.get(METADATA_BIN) or {}
        payload.update(metadata)
        return payload

    def _check_size(self, payload: dict, vector=None):
        size = len(json.dumps(payload or {}, default=str).encode("utf-8"))
        if vector is not None:
            size += len(vector) * 4
        if size > self.max_record_bytes:
            raise ValueError(f"record size {size} bytes exceeds max_record_bytes ({self.max_record_bytes})")

    # -- Insert, get, update, delete --------------------------------------------

    def insert(self, vectors, payloads=None, ids=None):
        payloads = payloads or [{} for _ in vectors]
        if ids is None:
            raise ValueError("ids is required for insert")
        if len(payloads) != len(vectors):
            raise ValueError(f"payloads length ({len(payloads)}) must match vectors length ({len(vectors)})")
        if len(ids) != len(vectors):
            raise ValueError(f"ids length ({len(ids)}) must match vectors length ({len(vectors)})")
        keys, bins_list = [], []
        for vector, payload, vector_id in zip(vectors, payloads, ids):
            self._check_size(payload, vector)
            keys.append(self.dataset.id(vector_id))
            bins_list.append(self._payload_to_bins(payload, vector))
        results = self.session.client.underlying_client.batch_write_blocking(
            keys, bins_list, batch_policy=self._batch_policy, write_policy=self._batch_write_policy
        )
        failures = [r for r in results if r.result_code != ResultCode.OK]
        if failures:
            first = failures[0]
            raise AerospikeError(
                f"batch insert failed for {len(failures)}/{len(results)} keys "
                f"(first: {first.key.value!r}: {first.server_message or first.result_code})",
                result_code=first.result_code,
                in_doubt=first.in_doubt,
            )

    def get(self, vector_id):
        key = self.dataset.id(vector_id)
        try:
            record = self.session.get(key)
        except AerospikeError as e:
            if e.result_code == ResultCode.KEY_NOT_FOUND_ERROR:
                return None
            raise
        return MemoryResult(id=vector_id, payload=self._bins_to_payload(record.bins))

    def delete(self, vector_id):
        key = self.dataset.id(vector_id)
        self.session.delete(key).execute()

    def update(self, vector_id, vector=None, payload=None):
        self._check_size(payload or {}, vector)
        bins = self._payload_to_bins(payload or {}, vector)
        key = self.dataset.id(vector_id)
        self.session.upsert(key).put(bins).execute()

    # -- Filtered access: shared scoping guard ----------------------------------

    def _resolve_scope(self, filters):
        """Return the scope field and value for the secondary-index Filter.

        Returns a tuple ``(scope_field, scope_value)``. ``scope_value`` is ``None``
        when a scope field is present but cannot be expressed as a single
        equality predicate (e.g. ``{"ne": "alice"}``). ``scope_field`` and
        ``scope_value`` are both ``None`` when no scope field is present.

        Raises when `filters` is non-empty but contains none of
        user_id/agent_id/run_id and `allow_scans_with_where` is not set.
        """
        if not filters:
            return None, None
        for field in SCOPE_FIELDS:
            if field not in filters:
                continue
            value = filters[field]
            if not isinstance(value, dict):
                return field, value
            if "eq" in value:
                return field, value["eq"]
        if any(field in filters for field in SCOPE_FIELDS):
            return None, None
        if not self.allow_scans_with_where:
            raise ValueError(
                "Filtered queries must include at least one of user_id, agent_id, or run_id as a "
                "scoping predicate, or set allow_scans_with_where=True to allow a full scan."
            )
        return None, None

    def _read_bin_names(self):
        return list(CORE_STRING_FIELDS) + ["created_at_ms", "updated_at_ms", METADATA_BIN]

    # -- list() ------------------------------------------------------------------

    def list(self, filters: dict = None, top_k: int = None):
        scope_field, scope_value = self._resolve_scope(filters)
        query = self.session.query(self.dataset)
        if scope_field is not None and scope_value is not None:
            query = query.filter(Filter.equal(scope_field, scope_value))
        remaining_expr = _compile_filters(filters, exclude_key=scope_field)
        if remaining_expr is not None:
            query = query.where(remaining_expr)

        query = query.bins(self._read_bin_names())
        if top_k:
            query = query.limit(top_k)

        results = []
        for row in query.execute():
            if not row.is_ok or row.record is None:
                continue
            payload = self._bins_to_payload(row.record.bins)
            results.append(MemoryResult(id=row.record.key.value, payload=payload))
        return [results]

    # -- search() ------------------------------------------------------------------

    def search(self, query, vectors, top_k: int = 5, filters: dict = None):
        scope_field, scope_value = self._resolve_scope(filters)
        qbuilder = self.session.query(self.dataset)
        if scope_field is not None and scope_value is not None:
            qbuilder = qbuilder.filter(Filter.equal(scope_field, scope_value))
        remaining_expr = _compile_filters(filters, exclude_key=scope_field)
        if remaining_expr is not None:
            qbuilder = qbuilder.where(remaining_expr)

        score_expr = Exp.cosine_similarity(Vector(vectors), Exp.vector_bin(EMBEDDING_BIN))
        ops = [Operation.get_bin(name) for name in self._read_bin_names()]
        ops.append(ExpOperation.read("score", score_expr))
        qbuilder = qbuilder.with_op_projection(*ops).order_by("score", OrderByType.DOUBLE, Order.DESC).top_k(top_k)

        results = []
        for row in qbuilder.execute():
            if not row.is_ok or row.record is None:
                continue
            bins = dict(row.record.bins)
            score = bins.pop("score", None)
            payload = self._bins_to_payload(bins)
            results.append(MemoryResult(id=row.record.key.value, payload=payload, score=score))
        return results

    def keyword_search(self, query: str, top_k: int = 5, filters: dict = None):
        return None
