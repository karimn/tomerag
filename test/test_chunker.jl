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

using TomeRAG: chunk_document, ChunkingConfig

@testset "chunk_document" begin
    md = """
    # Moves

    Intro paragraph.

    ## Delve the Depths

    **When you delve the depths**, roll +wits.

    ## Secure an Advantage

    **When you secure an advantage**, roll +heart.
    """
    cfg = ChunkingConfig(min_tokens=1, max_tokens=200, overlap_tokens=0)
    rawchunks = chunk_document(md, cfg)
    @test length(rawchunks) == 3
    @test rawchunks[1].heading_path == ["Moves"]
    @test rawchunks[2].heading_path == ["Moves", "Delve the Depths"]
    @test rawchunks[1].chunk_order == 0
    @test rawchunks[2].chunk_order == 1
    @test rawchunks[3].chunk_order == 2
end
