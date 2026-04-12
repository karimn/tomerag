module TomeRAG

include("content_types.jl")
include("types.jl")
include("tokenize.jl")
include("backends.jl")
include("chunker.jl")
include("storage.jl")
include("ingest.jl")
include("query_engine.jl")
include("server.jl")

export DEFAULT_CONTENT_TYPES, PBTA_CONTENT_TYPES, YZE_CONTENT_TYPES
export Chunk, QueryResult, ChunkingConfig
export Source, SourceRegistry, register_source!, get_source
export EmbeddingBackend, ClassifyBackend,
       MockEmbeddingBackend, MockClassifyBackend,
       OllamaBackend, HeuristicBackend,
       embed, classify
export split_to_token_budget
export initialize_store, insert_chunks, source_stats, similarity_search, bm25_search
export ingest!
export query, filter_chunks, lookup, multi_query, get_context, get_chunk
export serve

end # module
