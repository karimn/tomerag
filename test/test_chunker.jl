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

using TomeRAG: split_to_token_budget

@testset "split_to_token_budget — table kept atomic across token boundary" begin
    # Build a table + padding that exceeds max_tokens when combined.
    # The table alone is ~30 tokens; padded prefix is ~80 tokens → total > 100.
    table = """| Weapon      | Damage | Weight |
|-------------|--------|--------|
| Iron Sword  | 3      | 1      |
| Bone Spear  | 4      | 2      |
| Blight Blade| 5      | 3      |"""
    padding = join(fill("word", 80), " ")
    text = padding * "\n\n" * strip(table)

    pieces = split_to_token_budget(text; max_tokens=100, overflow=:paragraph)

    # The table must appear intact in exactly one piece (not split across pieces).
    pieces_with_table = filter(p -> occursin("Iron Sword", p), pieces)
    @test length(pieces_with_table) == 1          # appears in exactly one piece
    piece = pieces_with_table[1]
    @test occursin("Iron Sword", piece)
    @test occursin("Bone Spear", piece)
    @test occursin("Blight Blade", piece)          # all rows in the same piece
end

@testset "split_to_token_budget — table under max_tokens not split" begin
    table = """| A | B |
|---|---|
| 1 | 2 |
| 3 | 4 |"""
    # Table is small — should be returned as-is in one piece.
    pieces = split_to_token_budget(strip(table); max_tokens=200)
    @test length(pieces) == 1
    @test occursin("| 3 | 4 |", pieces[1])
end

@testset "split_to_token_budget — large table exceeding max kept whole" begin
    # A table larger than max_tokens should not be split (kept as one over-budget piece).
    rows = join(["| item_$(lpad(i, 3, '0')) | $(i*10) |" for i in 1:100], "\n")
    table = "| Item | Value |\n|------|-------|\n" * rows
    pieces = split_to_token_budget(table; max_tokens=50, overflow=:paragraph)
    # All rows must be in a single piece.
    @test length(filter(p -> occursin("item_001", p), pieces)) == 1
    @test length(filter(p -> occursin("item_100", p), pieces)) == 1
    piece_with_first = only(filter(p -> occursin("item_001", p), pieces))
    @test occursin("item_100", piece_with_first)
end
