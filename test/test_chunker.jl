using Test
using TomeRAG: parse_markdown_sections

@testset "parse_markdown_sections" begin
    md = """
    # Moves

    Intro paragraph.

    ## Delve the Depths

    **When you delve the depths**, roll +wits.

    On a 10+, choose two.

    ## Secure an Advantage

    **When you secure an advantage**, roll +heart.
    """
    secs = parse_markdown_sections(md)
    @test length(secs) == 3
    @test secs[1].heading_path == ["Moves"]
    @test occursin("Intro paragraph", secs[1].text)
    @test secs[2].heading_path == ["Moves", "Delve the Depths"]
    @test occursin("delve the depths", lowercase(secs[2].text))
    @test secs[3].heading_path == ["Moves", "Secure an Advantage"]
end
