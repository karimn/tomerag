using Test
using TomeRAG
using TomeRAG: extract_pages, extract_page, PageText

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
end
