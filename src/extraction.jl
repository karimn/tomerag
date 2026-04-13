using SHA
using Base64
using HTTP
using JSON3

# ── Abstract interface ─────────────────────────────────────────────────────────

abstract type ExtractionBackend end

"""
    PageText

One page of extracted text from a PDF, tagged with its 1-based page number.
"""
Base.@kwdef struct PageText
    page_num :: Int
    text     :: String
end

"""
    extract_pages(backend, pdf_path) -> Vector{PageText}

Extract all pages from `pdf_path` using `backend`. Returns one `PageText` per page,
ordered by page number.
"""
function extract_pages(::ExtractionBackend, path::AbstractString)
    error("extract_pages not implemented for this backend")
end

"""
    extract_page(backend, pdf_path, page_num) -> PageText

Extract a single page. Default implementation calls `extract_pages` and slices;
backends may override for efficiency.
"""
function extract_page(backend::ExtractionBackend, pdf_path::AbstractString, page_num::Int)
    pages = extract_pages(backend, pdf_path)
    idx = findfirst(p -> p.page_num == page_num, pages)
    idx === nothing && error("page $page_num not found in $pdf_path")
    return pages[idx]
end

# ── PopplerBackend ─────────────────────────────────────────────────────────────

const _PDFTOTEXT = Sys.which("pdftotext")
const _PDFTOPPM  = Sys.which("pdftoppm")
const _PDFINFO   = Sys.which("pdfinfo")

"""
    PopplerBackend()

Extract text with `pdftotext -layout` (from system Poppler installation). Fast and
free; degrades on multi-column layouts and loses table cell structure. Suitable for
single-column documents.
"""
struct PopplerBackend <: ExtractionBackend end

function _split_pdftext(output::AbstractString)
    raw_pages = split(output, '\f')
    result = PageText[]
    for (i, raw) in enumerate(raw_pages)
        text = strip(raw)
        isempty(text) || push!(result, PageText(page_num=i, text=String(text)))
    end
    return result
end

function extract_pages(::PopplerBackend, pdf_path::AbstractString)
    isnothing(_PDFTOTEXT) && error("pdftotext not found; install Poppler")
    output = readchomp(`$(_PDFTOTEXT) -layout $(pdf_path) -`)
    return _split_pdftext(output)
end

# Override for efficiency: extract one page with -f / -l flags instead of full PDF.
function extract_page(::PopplerBackend, pdf_path::AbstractString, page_num::Int)
    isnothing(_PDFTOTEXT) && error("pdftotext not found; install Poppler")
    output = readchomp(`$(_PDFTOTEXT) -layout -f $page_num -l $page_num $(pdf_path) -`)
    text = strip(output)
    isempty(text) && error("page $page_num is blank or out of range in $pdf_path")
    return PageText(page_num=page_num, text=String(text))
end

# ── CachingBackend ─────────────────────────────────────────────────────────────

"""
    CachingBackend(inner, cache_dir)

Transparent wrapper around any `ExtractionBackend`. Pages are saved to disk keyed
by `sha256(pdf_bytes)`, so the cache is valid regardless of filename. On re-ingest
of the same PDF, the inner backend is never called again.

Cache layout:
    cache_dir/<sha256_of_pdf>/page_001.txt
                              page_002.txt
                              ...

Note: The cache assumes inner backend returns pages with contiguous `page_num`
values starting at 1. `PopplerBackend` may skip blank pages (non-contiguous nums);
wrap it in a re-indexing step or use `MockExtractionBackend` for testing.
"""
struct CachingBackend <: ExtractionBackend
    inner     :: ExtractionBackend
    cache_dir :: String
end

function extract_pages(backend::CachingBackend, pdf_path::AbstractString)
    pdf_hash  = bytes2hex(sha256(read(pdf_path)))
    cache_dir = joinpath(backend.cache_dir, pdf_hash)

    # Try reading fully from cache (sequential page_NNN.txt files).
    if isdir(cache_dir) && isfile(joinpath(cache_dir, "page_001.txt"))
        cached = PageText[]
        i = 1
        while true
            f = joinpath(cache_dir, "page_$(lpad(i, 3, '0')).txt")
            isfile(f) || break
            push!(cached, PageText(page_num=i, text=read(f, String)))
            i += 1
        end
        isempty(cached) || return cached
    end

    # Cache miss — delegate to inner backend.
    results = extract_pages(backend.inner, pdf_path)

    mkpath(cache_dir)
    for pt in results
        write(joinpath(cache_dir, "page_$(lpad(pt.page_num, 3, '0')).txt"), pt.text)
    end
    return results
end

# ── VisionBackend ──────────────────────────────────────────────────────────────

const _VISION_PROMPT = """This is page {PAGE} of an RPG rulebook. Extract all text \
exactly as written. Format as markdown: use # headings for chapter/section titles, \
## for subsections, **bold** for move names and keywords, > blockquotes for sidebars \
and boxed text, and markdown tables for any tabular content. Preserve the reading \
order. Output only the extracted text, nothing else."""

"""
    VisionBackend(; model, api_key, concurrency, dpi)

Extract PDF pages by rendering each to PNG and sending the image to the Claude messages
API. Handles multi-column layouts, sidebars, and tables that defeat programmatic
extractors.

- `model`: Claude model ID (default: `"claude-haiku-4-5-20251001"`)
- `api_key`: Anthropic API key (default: `ENV["ANTHROPIC_API_KEY"]`)
- `concurrency`: max simultaneous API calls (default: `5`)
- `dpi`: page render resolution (default: `150` — sharp enough, token-efficient)
"""
Base.@kwdef struct VisionBackend <: ExtractionBackend
    model       :: String = "claude-haiku-4-5-20251001"
    api_key     :: String = get(ENV, "ANTHROPIC_API_KEY", "")
    concurrency :: Int    = 5
    dpi         :: Int    = 150
end

function _page_count(pdf_path::AbstractString)
    isnothing(_PDFINFO) && error("pdfinfo not found; install poppler-utils")
    output = readchomp(`$(_PDFINFO) $(pdf_path)`)
    m = match(r"Pages:\s+(\d+)", output)
    m === nothing && error("could not determine page count for $pdf_path")
    return parse(Int, m[1])
end

function _render_page(backend::VisionBackend, pdf_path::AbstractString, page_num::Int)
    isnothing(_PDFTOPPM) && error("pdftoppm not found; install poppler-utils")
    mktempdir() do tmpdir
        prefix = joinpath(tmpdir, "page")
        run(`$(_PDFTOPPM) -r $(backend.dpi) -f $page_num -l $page_num -png -singlefile \
             $pdf_path $prefix`)
        png_path = prefix * ".png"
        isfile(png_path) || error("pdftoppm did not produce $png_path for page $page_num")
        return read(png_path)
    end
end

function _call_vision_api(backend::VisionBackend, png_bytes::Vector{UInt8}, page_num::Int)
    encoded = base64encode(png_bytes)
    prompt  = replace(_VISION_PROMPT, "{PAGE}" => string(page_num))
    body = Dict(
        "model"      => backend.model,
        "max_tokens" => 4096,
        "messages"   => [Dict(
            "role"    => "user",
            "content" => [
                Dict("type" => "image", "source" => Dict(
                    "type"       => "base64",
                    "media_type" => "image/png",
                    "data"       => encoded,
                )),
                Dict("type" => "text", "text" => prompt),
            ],
        )],
    )
    resp = HTTP.post(
        "https://api.anthropic.com/v1/messages",
        ["x-api-key"         => backend.api_key,
         "anthropic-version" => "2023-06-01",
         "content-type"      => "application/json"],
        JSON3.write(body);
        readtimeout = 120,
    )
    HTTP.iserror(resp) &&
        error("Anthropic API error $(resp.status): $(String(resp.body))")
    result = JSON3.read(resp.body, Dict{String,Any})
    return String(result["content"][1]["text"])
end

# Single-page override — renders and calls API for one page only.
function extract_page(backend::VisionBackend, pdf_path::AbstractString, page_num::Int)
    png_bytes = _render_page(backend, pdf_path, page_num)
    text      = _call_vision_api(backend, png_bytes, page_num)
    return PageText(page_num=page_num, text=text)
end

# Full-document extraction using bounded concurrency.
function extract_pages(backend::VisionBackend, pdf_path::AbstractString)
    n = _page_count(pdf_path)
    results = Vector{Union{PageText,Nothing}}(nothing, n)
    sem     = Base.Semaphore(backend.concurrency)

    @sync for i in 1:n
        @async begin
            Base.acquire(sem)
            try
                results[i] = extract_page(backend, pdf_path, i)
            finally
                Base.release(sem)
            end
        end
    end
    return Vector{PageText}(results)
end
