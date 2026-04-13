using Test
using TomeRAG: extract_pages, extract_page, PageText, _split_pdftext, MockExtractionBackend, PopplerBackend

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
