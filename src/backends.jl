using SHA
using HTTP
using JSON3
using Preferences
using AnthropicSDK

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

# ---- API key management ------------------------------------------------------

"""
    _get_anthropic_key() -> String

Read the Anthropic API key. Preference `anthropic_api_key` (in `LocalPreferences.toml`)
takes priority over the `ANTHROPIC_API_KEY` environment variable.

To set via Preferences (recommended — survives shell restarts):
    using Preferences
    set_preferences!("TomeRAG", "anthropic_api_key" => "sk-ant-...")
"""
function _get_anthropic_key()
    k = @load_preference("anthropic_api_key", get(ENV, "ANTHROPIC_API_KEY", ""))
    isempty(k) && error(
        "Anthropic API key not set. Run:\n" *
        "  using Preferences\n" *
        "  set_preferences!(\"TomeRAG\", \"anthropic_api_key\" => \"sk-ant-...\")"
    )
    return k
end

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

# ---- Ollama embedding backend -----------------------------------------------

Base.@kwdef struct OllamaBackend <: EmbeddingBackend
    model::String
    base_url::String = "http://localhost:11434"
    dim::Int
    batch_size::Int = 32
end

function _ollama_embed(b::OllamaBackend, input)
    url = rstrip(b.base_url, '/') * "/api/embed"
    body = JSON3.write(Dict("model" => b.model, "input" => input))
    resp = HTTP.post(url, ["Content-Type" => "application/json"], body;
                     readtimeout=120, retry=false)
    resp.status == 200 || error("Ollama returned HTTP $(resp.status): $(String(resp.body))")
    payload = JSON3.read(resp.body)
    haskey(payload, :embeddings) || error("Ollama response missing 'embeddings' field")
    return [Vector{Float32}(e) for e in payload.embeddings]
end

function embed(b::OllamaBackend, text::AbstractString)
    v = _ollama_embed(b, [String(text)])
    return v[1]
end

function embed(b::OllamaBackend, texts::AbstractVector{<:AbstractString})
    out = Vector{Vector{Float32}}()
    sizehint!(out, length(texts))
    for i in 1:b.batch_size:length(texts)
        batch = texts[i:min(i + b.batch_size - 1, length(texts))]
        append!(out, _ollama_embed(b, collect(String, batch)))
    end
    return out
end

# ---- heuristic classify backend ----------------------------------------------

Base.@kwdef struct HeuristicBackend <: ClassifyBackend
    move_heading_pat::Regex    = r"moves?"i
    table_heading_pat::Regex   = r"tables?|oracle|random"i
    bestiary_heading_pat::Regex = r"bestiary|stat ?block|npcs?|monsters?"i
    lore_heading_pat::Regex    = r"world|geography|history|cult|faction|lore"i
    gm_heading_pat::Regex      = r"running|gm ?(guide|advice)|how to run"i
end

const _MOVE_TRIGGER_PAT = r"\*\*when\s+([^*]+?)\*\*"i
const _TABLE_PAT        = r"^\s*\|.*\|\s*$"m
const _STAT_LINE_PAT    = r"\b(hp|hit points|armor|ac|attack)\b[^\n]*\d"i

function classify(h::HeuristicBackend; text, heading_path)
    joined_heading = lowercase(join(heading_path, " / "))

    # 1. PbtA move — bold **When ...** trigger in the body (strongest signal)
    m = match(_MOVE_TRIGGER_PAT, text)
    if m !== nothing
        trigger = strip(String(m.captures[1]))
        return (content_type=:move, tags=String[], move_trigger=trigger,
                scene_type=nothing, encounter_key=nothing, npc_name=nothing)
    end

    # 2. Table — markdown pipe rows OR oracle-ish heading
    if count(_TABLE_PAT, text) >= 2 || occursin(h.table_heading_pat, joined_heading)
        return (content_type=:table, tags=String[], move_trigger=nothing,
                scene_type=nothing, encounter_key=nothing, npc_name=nothing)
    end

    # 3. Stat block — bestiary heading + stat-like lines
    if occursin(h.bestiary_heading_pat, joined_heading) && occursin(_STAT_LINE_PAT, text)
        name = isempty(heading_path) ? nothing : String(strip(last(heading_path)))
        return (content_type=:stat_block, tags=String[], move_trigger=nothing,
                scene_type=nothing, encounter_key=nothing, npc_name=name)
    end

    # 4. GM guidance
    if occursin(h.gm_heading_pat, joined_heading)
        return (content_type=:gm_guidance, tags=String[], move_trigger=nothing,
                scene_type=nothing, encounter_key=nothing, npc_name=nothing)
    end

    # 5. Lore
    if occursin(h.lore_heading_pat, joined_heading)
        return (content_type=:lore, tags=String[], move_trigger=nothing,
                scene_type=nothing, encounter_key=nothing, npc_name=nothing)
    end

    # 6. Default: mechanic
    return (content_type=:mechanic, tags=String[], move_trigger=nothing,
            scene_type=nothing, encounter_key=nothing, npc_name=nothing)
end

# ---- mock extraction backend -------------------------------------------------

"""
    MockExtractionBackend(pages)

Returns the given `Vector{PageText}` verbatim, ignoring the pdf_path.
Use in tests to avoid real PDF files or API calls.
"""
struct MockExtractionBackend <: ExtractionBackend
    pages::Vector{PageText}
end

extract_pages(b::MockExtractionBackend, ::AbstractString) = b.pages
