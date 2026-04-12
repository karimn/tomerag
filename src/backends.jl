using SHA

# ---- abstract types ---------------------------------------------------------

abstract type EmbeddingBackend end
abstract type ClassifyBackend end

"""
    embed(backend, text) -> Vector{Float32}
    embed(backend, texts::Vector{<:AbstractString}) -> Vector{Vector{Float32}}

Produce dense embeddings. Backends must implement both methods.
"""
function embed end

"""
    classify(backend; text, heading_path) -> NamedTuple

Returns `(content_type::Symbol, tags::Vector{String},
          move_trigger::Union{String,Nothing}, scene_type::Union{Symbol,Nothing},
          encounter_key::Union{String,Nothing}, npc_name::Union{String,Nothing})`.
"""
function classify end

# ---- mock embedding backend -------------------------------------------------

struct MockEmbeddingBackend <: EmbeddingBackend
    dim::Int
end
MockEmbeddingBackend(; dim::Int=8) = MockEmbeddingBackend(dim)

function embed(b::MockEmbeddingBackend, text::AbstractString)
    # Deterministic pseudo-embedding derived from SHA256 of the text.
    h = sha256(codeunits(text))
    v = Vector{Float32}(undef, b.dim)
    @inbounds for i in 1:b.dim
        byte = h[((i - 1) % length(h)) + 1]
        v[i] = Float32((byte / 255.0) * 2 - 1)   # in [-1, 1]
    end
    # L2 normalize so cosine distance behaves.
    n = sqrt(sum(x -> x * x, v))
    return n == 0 ? v : v ./ Float32(n)
end

embed(b::MockEmbeddingBackend, texts::AbstractVector{<:AbstractString}) =
    [embed(b, t) for t in texts]

# ---- mock classify backend --------------------------------------------------

Base.@kwdef struct MockClassifyBackend <: ClassifyBackend
    content_type::Symbol = :mechanic
    tags::Vector{String} = String[]
end

function classify(b::MockClassifyBackend; text, heading_path)
    return (
        content_type  = b.content_type,
        tags          = copy(b.tags),
        move_trigger  = nothing,
        scene_type    = nothing,
        encounter_key = nothing,
        npc_name      = nothing,
    )
end
