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
function extract_pages end

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
