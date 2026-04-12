# TomeRAG.jl

A Julia library for building RAG pipelines specialized for tabletop RPG content —
rules, lore, campaigns, and adventures. One `.duckdb` per RPG system, pluggable
embedding and classification backends, heading-aware chunking.

**Status:** Plan 1 MVP — markdown ingestion + semantic search only. REST API,
hybrid BM25+dense retrieval, and PDF ingestion land in later plans.

## Install

```julia
] dev /path/to/TomeRAG.jl
```

Requires Julia 1.10+. Ollama is optional; tests use a mock backend.

## Usage

```julia
using TomeRAG

src = Source(
    id = "ironsworn",
    name = "Ironsworn",
    system = "PbtA",
    db_path = "sources/ironsworn.duckdb",
    embedding_model = "nomic-embed-text",
    embedding_dim = 768,
    license = :cc_by,
    chunking = ChunkingConfig(),
    content_types = PBTA_CONTENT_TYPES,
)

reg = SourceRegistry()
register_source!(reg, src)
initialize_store(src)

backend = OllamaBackend(model="nomic-embed-text", dim=768)

ingest!(reg, "ironsworn", "path/to/ironsworn.md";
        doc_id = "ironsworn-core",
        document_type = :core_rules,
        format = :markdown,
        embed_backend = backend,
        classify_backend = HeuristicBackend())

results = query(reg, "How does momentum work on a miss?";
                source = "ironsworn",
                content_type = :move,
                top_k = 5,
                embed_backend = backend)

for r in results
    println(r.rank, "  ", r.score, "  ", join(r.chunk.heading_path, " / "))
    println("  ", first(r.chunk.text, 120), "...")
end
```

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'

# Optional: run Ollama live embedding tests
TOMERAG_TEST_OLLAMA=1 julia --project=. -e 'using Pkg; Pkg.test()'
```
