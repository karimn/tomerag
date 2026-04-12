"""
    Section

Internal: a heading-delimited span of markdown with its heading path.
"""
Base.@kwdef struct Section
    heading_path::Vector{String}
    text::String
end

"""
    parse_markdown_sections(md) -> Vector{Section}

Walks markdown by `#`-headings, emitting one Section per heading with the body
that follows until the next heading. Heading path is the stack of ancestors.
"""
function parse_markdown_sections(md::AbstractString)
    sections = Section[]
    stack = String[]        # current heading path
    buf = IOBuffer()
    started = false         # have we seen any heading yet?

    function flush!()
        if started
            t = strip(String(take!(buf)))
            if !isempty(t)
                push!(sections, Section(heading_path=copy(stack), text=t))
            else
                take!(buf)
            end
        else
            take!(buf)      # discard preamble before any heading
        end
    end

    for line in eachline(IOBuffer(md))
        m = match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m !== nothing
            flush!()
            level = length(m.captures[1])
            title = String(m.captures[2])
            # Pop deeper levels.
            while length(stack) >= level
                pop!(stack)
            end
            # Pad up to level-1, then push this title at level.
            while length(stack) < level - 1
                push!(stack, "")
            end
            push!(stack, title)
            started = true
        else
            println(buf, line)
        end
    end
    flush!()
    return sections
end

"""
    split_to_token_budget(text; max_tokens, overlap_tokens, overflow) -> Vector{String}

If text fits in `max_tokens`, returns `[text]`. Otherwise applies the overflow
ladder: `:paragraph` → `:sentence` → `:token`. `overlap_tokens` of trailing
tokens from one piece are prepended to the next.
"""
function split_to_token_budget(text::AbstractString; max_tokens::Int,
                               overlap_tokens::Int=0, overflow::Symbol=:paragraph)
    toks = split(text)
    if length(toks) <= max_tokens
        return [String(text)]
    end

    # Try paragraph splits first.
    if overflow == :paragraph
        paras = split(text, r"\n\s*\n")
        pieces = _greedy_pack(paras, max_tokens)
        if all(length(split(p)) <= max_tokens for p in pieces)
            return _apply_overlap(pieces, overlap_tokens)
        end
        overflow = :sentence
    end

    if overflow == :sentence
        sents = split(text, r"(?<=[.!?])\s+")
        pieces = _greedy_pack(sents, max_tokens)
        if all(length(split(p)) <= max_tokens for p in pieces)
            return _apply_overlap(pieces, overlap_tokens)
        end
        overflow = :token
    end

    # Hard token cap.
    pieces = String[]
    i = 1
    while i <= length(toks)
        j = min(i + max_tokens - 1, length(toks))
        push!(pieces, join(toks[i:j], " "))
        i = j + 1
    end
    return _apply_overlap(pieces, overlap_tokens)
end

function _greedy_pack(units::AbstractVector, max_tokens::Int)
    out = String[]
    cur = IOBuffer()
    cur_tokens = 0
    for u in units
        u_str = strip(String(u))
        isempty(u_str) && continue
        u_tokens = length(split(u_str))
        if cur_tokens + u_tokens > max_tokens && cur_tokens > 0
            push!(out, strip(String(take!(cur))))
            cur_tokens = 0
        end
        if u_tokens > max_tokens
            cur_tokens > 0 && push!(out, strip(String(take!(cur))))
            push!(out, u_str)
            cur_tokens = 0
        else
            cur_tokens > 0 && print(cur, " ")
            print(cur, u_str)
            cur_tokens += u_tokens
        end
    end
    cur_tokens > 0 && push!(out, strip(String(take!(cur))))
    return out
end

function _apply_overlap(pieces::Vector{String}, overlap_tokens::Int)
    overlap_tokens <= 0 && return pieces
    length(pieces) <= 1 && return pieces
    out = Vector{String}(undef, length(pieces))
    out[1] = pieces[1]
    for i in 2:length(pieces)
        prev_toks = split(pieces[i-1])
        tail = prev_toks[max(end - overlap_tokens + 1, 1):end]
        out[i] = join(tail, " ") * " " * pieces[i]
    end
    return out
end

"""
    RawChunk

Intermediate chunk: text, heading path, chunk_order within the document,
page (empty string for markdown — set by PDF extractor later).
"""
Base.@kwdef struct RawChunk
    heading_path::Vector{String}
    text::String
    chunk_order::Int
    page::String = ""
end

"""
    chunk_document(md, cfg) -> Vector{RawChunk}

Markdown → section parse → token-budget split → sequential numbering.
Sections with fewer than `cfg.min_tokens` tokens are merged into the next section.
"""
function chunk_document(md::AbstractString, cfg::ChunkingConfig)
    sections = parse_markdown_sections(md)
    out = RawChunk[]
    pending = nothing   # Section pending merge due to min_tokens

    function emit(sec::Section)
        pieces = split_to_token_budget(sec.text;
            max_tokens     = cfg.max_tokens,
            overlap_tokens = cfg.overlap_tokens,
            overflow       = cfg.overflow)
        for p in pieces
            push!(out, RawChunk(
                heading_path = sec.heading_path,
                text         = p,
                chunk_order  = length(out),
            ))
        end
    end

    for sec in sections
        tc = length(split(sec.text))
        if tc < cfg.min_tokens
            if pending === nothing
                pending = sec
            else
                pending = Section(
                    heading_path = pending.heading_path,
                    text         = pending.text * "\n\n" * sec.text)
            end
            continue
        end
        if pending !== nothing
            merged = Section(
                heading_path = pending.heading_path,
                text         = pending.text * "\n\n" * sec.text)
            emit(merged)
            pending = nothing
        else
            emit(sec)
        end
    end
    pending !== nothing && emit(pending)
    return out
end
