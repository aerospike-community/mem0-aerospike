# mem0 vector-store benchmarks

Benchmarks for three mem0 vector-store providers that share the same workload
shape, for direct comparison:

- `mem0.vector_stores.redis.RedisDB` — see `mem0_redis_vector_workload.py`
- `mem0.vector_stores.aerospike.AerospikeDB` — see `mem0_aerospike_vector_workload.py`
- `mem0.vector_stores.pgvector.PGVector` — see `mem0_postgres_vector_workload.py`

Both workloads exercise the public `VectorStoreBase` interface
(`insert`, `get`, `search`, `list`, `update`) and are driven by the same
`ai-ecosystem-benchmark` `BenchmarkRunner`. `delete` is intentionally out of
scope for both workloads — see the module docstrings for the rationale.

## Comparability caveat: exact search, not ANN

`RedisDB` configures RediSearch's `FLAT` algorithm (exact, brute-force cosine
similarity), `AerospikeDB` uses Aerospike's native `Vector` bin with
`Exp.cosine_similarity` plus server-side Top-K, and `PGVector` is created with
`hnsw=False`/`diskann=False` so it performs exact `<=>` (cosine-distance)
ordering over the filtered candidate set. None of the three is an ANN index, so
the results are comparable as exact-search measurements. Every result records
`index_algorithm: "FLAT"` and `distance_metric: "cosine"` in its metadata.

Exact search cost scales with the scanned candidate set, so results are only
comparable at a fixed `corpus_size` and `embedding_model_dims` (recorded in every
result's metadata). Treat different dimensions as separate sweep points, not one
number.

## Scenarios

Each workload provides five scenarios, prefixed with the backend name so the
runner discovers them:

- `<backend>_insert` — inserts exactly one new vector/payload/id per call.
- `<backend>_get` — point lookup of a corpus record by id.
- `<backend>_search` — vector search with a `user_id` filter.
- `<backend>_list` — filtered list with a `user_id` filter.
- `<backend>_update` — full replace of an existing corpus record's vector and payload.

All scenarios share one collection and one pre-seeded read corpus. No scenario
removes read-corpus records, so scenarios can run in any order (the runner
discovers them alphabetically) without one scenario silently biasing another's
measurements. `setup()` truncates and fully re-seeds the collection at the start
of every run, so a prior interrupted run can never leak state into a new one.

## Local development

The benchmarks run against local containers — no managed service, cloud account,
or chargeable infrastructure is required.

```shell
cd mem0-aerospike/benchmarks
docker compose up -d --wait
```

This starts:

- Redis Stack (`redis/redis-stack-server:7.4.0-v3`) on `:6379`, bundling the
  RediSearch and RedisJSON modules `RedisDB` requires.
- Aerospike Server (`aerospike/aerospike-server:latest`) on `:3000-3002`.
  The `AerospikeDB` provider requires Aerospike Server 8.1.3+ (8.2.0+
  recommended).
- PostgreSQL with pgvector (`pgvector/pgvector:pg16`) on `:5432`.

Install the `benchmark` optional-dependency group alongside `vector-stores`:

```shell
cd mem0-aerospike
uv pip install -e ".[vector-stores,benchmark,test]"
```

### Run the Redis workload

```shell
python -m benchmarks.mem0_redis_vector_workload \
  --redis-url redis://localhost:6379 \
  --qps 50 --warmup-seconds 1 --duration-seconds 10 \
  --output mem0-redis-vector-benchmark.json
```

### Run the Aerospike workload

```shell
python -m benchmarks.mem0_aerospike_vector_workload \
  --aerospike-connection-string localhost:3000:test \
  --qps 50 --warmup-seconds 1 --duration-seconds 10 \
  --output mem0-aerospike-vector-benchmark.json
```

### Run the Postgres/pgvector workload

```shell
python -m benchmarks.mem0_postgres_vector_workload \
  --postgres-connection-string postgresql://postgres:postgres@localhost:5432/postgres \
  --qps 50 --warmup-seconds 1 --duration-seconds 10 \
  --output mem0-postgres-vector-benchmark.json
```

Key parameters (see `WorkloadConfig` for the full list and defaults):

| Flag | Meaning |
| --- | --- |
| `--seed` | Random seed; same seed + parameters reproduce the same corpus and per-call sampling sequence |
| `--embedding-dims` | Embedding dimensionality |
| `--corpus-size` | Number of records pre-seeded for `get`/`search`/`list`/`update` |
| `--key-pool-size` | Number of distinct `user_id` values assigned round-robin across the corpus |
| `--payload-bytes` | Size of each record's generated payload text |
| `--top-k` | Result limit for `search`/`list` |
| `--qps` | Offered queries per second per scenario |

Results are written to the `--output` JSON path and include response/service
latency percentiles, achieved QPS, failures, workload parameters, the index
algorithm and distance metric, and `mem0ai`/`ai-ecosystem-benchmark` package
versions. No connection string or credential ever appears in the output.

## Running tests

Unit tests use in-memory fakes for each provider and require no external
services. A smoke test for each workload additionally runs all five scenarios
against a real local instance and is skipped automatically if the corresponding
service is not reachable.

```shell
docker compose -f benchmarks/docker-compose.yml up -d --wait
pytest tests/benchmarks/
```
