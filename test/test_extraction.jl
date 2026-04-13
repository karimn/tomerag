using Test
using TomeRAG: extract_pages, extract_page, PageText, _split_pdftext, MockExtractionBackend, PopplerBackend, CachingBackend, ExtractionBackend

@testset "MockExtractionBackend" begin
    pages = [PageText(1, "# Iron Vow\nRoll +heart."),
             PageText(2, "# Face Danger\nRoll +edge.")]
    b = MockExtractionBackend(pages)

    result = extract_pages(b, "any/path.pdf")
    @test length(result) == 2
    @test result[1].page_num == 1
    @test result[1].text == "# Iron Vow\nRoll +heart."
    @test result[2].page_num == 2
    @test result[2].text == "# Face Danger\nRoll +edge."

    # Default extract_page delegates to extract_pages
    p = extract_page(b, "any/path.pdf", 2)
    @test p.page_num == 2
    @test p.text == "# Face Danger\nRoll +edge."

    # extract_page raises an error when page_num is not in results
    @test_throws ErrorException extract_page(b, "any/path.pdf", 99)
end

@testset "PopplerBackend — _split_pdftext unit" begin
    # Form-feed (\f) separates pages in pdftotext output
    output = "Page one content.\fPage two content.\f"
    pages = _split_pdftext(output)
    @test length(pages) == 2
    @test pages[1].page_num == 1
    @test pages[1].text == "Page one content."
    @test pages[2].page_num == 2
    @test pages[2].text == "Page two content."
end

@testset "PopplerBackend — _split_pdftext skips blank pages" begin
    output = "Content\f\f\fMore content"   # two empty pages in the middle
    pages = _split_pdftext(output)
    @test length(pages) == 2
    @test pages[1].text == "Content"
    @test pages[2].text == "More content"
    # page_num reflects original position (3 was skipped, 4 is the second result)
    @test pages[1].page_num == 1
    @test pages[2].page_num == 4
end

@testset "PopplerBackend — extract_page uses -f/-l flags" begin
    # Live test: only runs if TOMERAG_LIVE_TESTS=1 and test fixture exists
    if get(ENV, "TOMERAG_LIVE_TESTS", "0") == "1"
        fixture = joinpath(@__DIR__, "fixtures", "test.pdf")
        if isfile(fixture)
            b = PopplerBackend()
            p = extract_page(b, fixture, 1)
            @test p.page_num == 1
            @test !isempty(p.text)

            pages = extract_pages(b, fixture)
            @test length(pages) >= 1
            @test all(p -> !isempty(p.text), pages)
        else
            @warn "Skipping PopplerBackend live test: test/fixtures/test.pdf not found"
        end
    end
end

@testset "CachingBackend — first call extracts and caches" begin
    tmpdir = mktempdir()
    pdf_path = joinpath(tmpdir, "test.pdf")
    write(pdf_path, "fake pdf bytes for hashing")

    pages = [PageText(page_num=1, text="iron vow text"),
             PageText(page_num=2, text="face danger text")]
    inner = MockExtractionBackend(pages)
    cache = CachingBackend(inner, joinpath(tmpdir, "cache"))

    r1 = extract_pages(cache, pdf_path)
    @test length(r1) == 2
    @test r1[1].page_num == 1
    @test r1[1].text == "iron vow text"

    # Cache files must exist on disk
    using SHA
    pdf_hash = bytes2hex(sha256(read(pdf_path)))
    cache_dir = joinpath(tmpdir, "cache", pdf_hash)
    @test isfile(joinpath(cache_dir, "page_001.txt"))
    @test isfile(joinpath(cache_dir, "page_002.txt"))
    @test read(joinpath(cache_dir, "page_001.txt"), String) == "iron vow text"
end

@testset "CachingBackend — second call returns from disk (inner not called again)" begin
    tmpdir = mktempdir()
    pdf_path = joinpath(tmpdir, "test.pdf")
    write(pdf_path, "stable content for hash")

    call_count = Ref(0)

    # Counting mock — increments counter on each extract_pages call
    struct _CountingMock <: ExtractionBackend
        pages::Vector{PageText}
        counter::Ref{Int}
    end
    TomeRAG.extract_pages(b::_CountingMock, ::AbstractString) = (b.counter[] += 1; b.pages)

    pages = [PageText(page_num=1, text="cached text")]
    inner = _CountingMock(pages, call_count)
    cache = CachingBackend(inner, joinpath(tmpdir, "cache"))

    extract_pages(cache, pdf_path)          # populates cache
    @test call_count[] == 1

    extract_pages(cache, pdf_path)          # should read from disk
    @test call_count[] == 1                 # NOT incremented again
end

@testset "CachingBackend — different PDF content → different cache dir" begin
    tmpdir = mktempdir()
    p1 = joinpath(tmpdir, "a.pdf"); write(p1, "pdf content A")
    p2 = joinpath(tmpdir, "b.pdf"); write(p2, "pdf content B")

    pages = [PageText(page_num=1, text="content")]
    inner = MockExtractionBackend(pages)
    cache = CachingBackend(inner, joinpath(tmpdir, "cache"))

    extract_pages(cache, p1)
    extract_pages(cache, p2)

    using SHA
    h1 = bytes2hex(sha256(read(p1)))
    h2 = bytes2hex(sha256(read(p2)))
    @test h1 != h2
    @test isdir(joinpath(tmpdir, "cache", h1))
    @test isdir(joinpath(tmpdir, "cache", h2))
end
