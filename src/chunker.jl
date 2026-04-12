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
