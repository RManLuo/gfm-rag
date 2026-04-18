# ColbertELModel Replacement Design

Date: 2026-04-18

## Goal

Replace the current `colbert-ai`-based implementation inside `gfmrag.graph_index_construction.entity_linking_model.ColbertELModel` with a `Qdrant + FastEmbed` implementation while keeping the public behavior as close as practical to the existing model.

The replacement must preserve:

- The public class name: `ColbertELModel`
- The public interface: `__init__(...)`, `index(entity_list)`, `__call__(ner_entity_list, topk)`
- The current return schema from `__call__`
- The current cache reuse semantics based on the entity-list fingerprint
- The current `processing_phrases(...)` preprocessing path

The replacement does not try to preserve bit-identical scores or rankings. The goal is interface compatibility, similar retrieval behavior, and removal of the `colbert-ai` dependency.

## Current Constraints

The existing implementation:

- Uses `processing_phrases(...)` before indexing and querying
- Computes a fingerprint with `md5("".join(entity_list).encode()).hexdigest()`
- Stores index artifacts under `root/colbert/{fingerprint}`
- Reuses cached index data when `force=False`
- Rebuilds from scratch when `force=True`
- Returns results as:

```python
{
    query: [
        {
            "entity": original_entity,
            "score": raw_score,
            "norm_score": raw_score / top1_score,
        },
        ...,
    ]
}
```

The new design keeps these semantics unless explicitly noted otherwise.

## Proposed Architecture

`ColbertELModel` remains the exported class. Its internals change from:

- `colbert.Indexer`
- `colbert.Searcher`
- `colbert.data.Queries`
- `colbert.infra.*`

to:

- `qdrant_client.QdrantClient`
- `qdrant_client.models`
- `fastembed` ColBERT multivector embeddings

The model becomes a thin adapter with three responsibilities:

1. Preprocess entity and query text with the existing phrase normalization
2. Manage a Qdrant collection lifecycle keyed by the existing fingerprint
3. Convert Qdrant query results back into the current output format

## Public API

The constructor keeps the current parameters for compatibility:

- `model_name_or_path: str = "colbert-ir/colbertv2.0"`
- `root: str = "tmp"`
- `doc_index_name: str = "nbits_2"`
- `phrase_index_name: str = "nbits_2"`
- `force: bool = False`

The constructor also adds one new optional runtime parameter:

- `use_in_memory: bool = False`

Behavior:

- `use_in_memory=False`:
  - Use local persistent Qdrant storage under `root/colbert/{fingerprint}`
  - Enable cross-process cache reuse
- `use_in_memory=True`:
  - Use `QdrantClient(":memory:")`
  - Do not create persistent Qdrant files
  - Do not create metadata files
  - Do not support cross-process cache reuse

The existing `doc_index_name` remains accepted even though only phrase/entity retrieval is used. This avoids breaking Hydra configs or call sites that still pass it.

## Internal State

The replacement model keeps these runtime fields:

- `self.model_name_or_path`
- `self.root`
- `self.doc_index_name`
- `self.phrase_index_name`
- `self.force`
- `self.use_in_memory`
- `self.entity_list`
- `self.client`
- `self.collection_name`
- `self.collection_root`
- `self.metadata_path`
- `self.embedder`

## Data Flow

### `index(entity_list)`

1. Save `self.entity_list`
2. Compute `fingerprint = md5("".join(entity_list).encode()).hexdigest()`
3. Build `collection_root = root/colbert/{fingerprint}` when `use_in_memory=False`
4. Preprocess entity strings with `processing_phrases(...)`
5. If persistent mode is enabled and reusable cached artifacts exist, attach to the cached collection and return
6. Otherwise:
   - initialize the FastEmbed ColBERT embedder
   - encode processed entities into multivectors
   - create a Qdrant collection configured for multivector `MAX_SIM`
   - insert one point per entity
   - write metadata when persistent mode is enabled

Each inserted point stores:

- `id`: entity index
- `vector`: entity multivector
- `payload.entity`: original entity string
- `payload.processed_entity`: normalized entity string

### `__call__(ner_entity_list, topk)`

1. Fail if `index(...)` has not been called yet
2. Preprocess query strings with `processing_phrases(...)`
3. Encode each query into a query multivector using the same embedder
4. Query the Qdrant collection with `MAX_SIM`
5. Convert the top-k hits into the existing result schema

`norm_score` remains:

```python
score / top1_score if top1_score else 0.0
```

## Qdrant Collection Design

Each collection uses a single multivector field with:

- distance: cosine
- multivector comparator: `MAX_SIM`

The collection name should be deterministic and local to this model instance, for example:

- `phrase_index_name`

Because storage is already isolated under `root/colbert/{fingerprint}` in persistent mode, reusing `phrase_index_name` inside that local store is acceptable and keeps the design close to the current implementation.

## Cache Reuse Semantics

Persistent mode intentionally mirrors the current `ColbertELModel` behavior.

### Persistent mode

Storage path:

- `root/colbert/{fingerprint}/`

Artifacts in that directory:

- Qdrant local data
- `metadata.json`

Reuse is allowed only when all of the following are true:

- `force=False`
- the storage directory exists
- the Qdrant collection exists
- `metadata.json` exists
- the metadata matches the current runtime configuration

### In-memory mode

When `use_in_memory=True`:

- no `root/colbert/{fingerprint}` directory is created
- no `metadata.json` is written
- no cross-process cache reuse is attempted

This is intentionally an all-in-memory mode, not a hybrid mode.

## Metadata Contract

Persistent mode writes `metadata.json` with enough information to prevent invalid reuse.

Required fields:

- `fingerprint`
- `model_name_or_path`
- `backend = "qdrant_fastembed_colbert"`
- `phrase_index_name`
- `processed_entities_hash`
- `preprocess_version`
- `multivector_dim`

Cache must be treated as stale and rebuilt if any required field is missing or mismatched.

This is stricter than the old implementation and is intentional. The old implementation keyed only by entity content. The new implementation avoids silently reusing incompatible data after model or preprocessing changes.

## Rebuild Rules

Rebuild the collection when any of the following are true:

- `force=True`
- metadata file is missing
- metadata does not match the current configuration
- collection is missing or unreadable
- Qdrant local storage is corrupted or incomplete

In persistent mode, rebuild means removing `root/colbert/{fingerprint}` and creating it again.

## Behavior Compatibility

The design intentionally preserves:

- class name
- constructor surface
- call signatures
- preprocessing path
- output schema
- top-score normalization logic
- fingerprint-driven cache reuse pattern

The design does not promise:

- identical raw scores
- identical ranking ties
- identical internal indexing artifacts

This is acceptable because the backend changes from Stanford ColBERT indexing code to Qdrant multivector search, while the retrieval paradigm remains late interaction with `MAX_SIM`.

## Dependency Changes

Add the Qdrant/FastEmbed dependencies needed for:

- local and in-memory Qdrant usage
- ColBERT multivector embedding generation

Remove the direct runtime dependency on `colbert-ai` after the new implementation is complete and verified.

## Testing Strategy

### Unit and smoke tests

Keep `tests/test_el_model.py` exercising:

- Hydra instantiation of `ColbertELModel`
- `index(entity_list)`
- `__call__(queries, topk)`
- result shape is a `dict`

Add focused tests for:

- persistent cache reuse with the same `entity_list`
- rebuild when `force=True`
- in-memory mode does not create persistent files

### Lightweight verification

Run:

```bash
python -m py_compile gfmrag/graph_index_construction/entity_linking_model/colbert_el_model.py
```

Then run the repo-preferred hook suite:

```bash
conda activate gfmrag2
pre-commit run --all-files
```

## Non-Goals

- Reproducing Stanford ColBERT indexing byte-for-byte
- Keeping the old on-disk artifact format
- Adding training support for ColBERT
- Refactoring unrelated EL models

## Implementation Notes

- The replacement should be implemented in the existing `colbert_el_model.py` file rather than by introducing a parallel public model first.
- Keep comments minimal and only where the cache lifecycle or Qdrant setup would otherwise be hard to parse.
- Prefer deterministic naming and deterministic cache checks so failures are easy to debug.
