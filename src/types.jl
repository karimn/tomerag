Base.@kwdef struct ChunkingConfig
    min_tokens::Int = 100
    max_tokens::Int = 800
    overflow::Symbol = :paragraph          # :paragraph | :sentence | :token
    overlap_tokens::Int = 50
    atomic_patterns::Vector{Regex} = Regex[]
end

Base.@kwdef struct Chunk
    id::String
    source_id::String
    doc_id::String
    doc_path::String

    text::String
    embedding::Vector{Float32}
    embedding_model::String
    token_count::Int
    content_hash::String

    document_type::Symbol
    system::String
    edition::String
    page::String
    heading_path::Vector{String}
    chunk_order::Int
    parent_id::Union{String,Nothing}

    content_type::Symbol
    tags::Vector{String}

    move_trigger::Union{String,Nothing}
    scene_type::Union{Symbol,Nothing}
    encounter_key::Union{String,Nothing}
    npc_name::Union{String,Nothing}

    license::Symbol
end

struct QueryResult
    chunk::Chunk
    score::Float32
    rank::Int
end
