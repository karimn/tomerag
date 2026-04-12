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

# Copy-with-keyword-override constructor for Chunk.
function Chunk(c::Chunk;
               id=c.id, source_id=c.source_id, doc_id=c.doc_id, doc_path=c.doc_path,
               text=c.text, embedding=c.embedding, embedding_model=c.embedding_model,
               token_count=c.token_count, content_hash=c.content_hash,
               document_type=c.document_type, system=c.system, edition=c.edition,
               page=c.page, heading_path=c.heading_path, chunk_order=c.chunk_order,
               parent_id=c.parent_id, content_type=c.content_type, tags=c.tags,
               move_trigger=c.move_trigger, scene_type=c.scene_type,
               encounter_key=c.encounter_key, npc_name=c.npc_name, license=c.license)
    return Chunk(id=id, source_id=source_id, doc_id=doc_id, doc_path=doc_path,
                 text=text, embedding=embedding, embedding_model=embedding_model,
                 token_count=token_count, content_hash=content_hash,
                 document_type=document_type, system=system, edition=edition,
                 page=page, heading_path=heading_path, chunk_order=chunk_order,
                 parent_id=parent_id, content_type=content_type, tags=tags,
                 move_trigger=move_trigger, scene_type=scene_type,
                 encounter_key=encounter_key, npc_name=npc_name, license=license)
end

Base.@kwdef struct Source
    id::String
    name::String
    system::String
    db_path::String
    embedding_model::String
    embedding_dim::Int
    license::Symbol
    chunking::ChunkingConfig
    content_types::Set{Symbol}
end

struct SourceRegistry
    sources::Dict{String,Source}
end
SourceRegistry() = SourceRegistry(Dict{String,Source}())

function register_source!(reg::SourceRegistry, s::Source)
    haskey(reg.sources, s.id) && error("source id already registered: $(s.id)")
    reg.sources[s.id] = s
    return s
end

function get_source(reg::SourceRegistry, id::AbstractString)
    haskey(reg.sources, id) || throw(KeyError(id))
    return reg.sources[id]
end
