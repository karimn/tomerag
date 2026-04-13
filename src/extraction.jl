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
