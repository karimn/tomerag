using Test
using TomeRAG: split_to_token_budget

@testset "split_to_token_budget" begin
    # Below budget -> single piece.
    parts = split_to_token_budget("one two three", max_tokens=10, overlap_tokens=2,
                                  overflow=:paragraph)
    @test parts == ["one two three"]

    # Paragraph split
    txt = "para one has several words here.\n\npara two has several words too.\n\npara three closes it out."
    parts2 = split_to_token_budget(txt, max_tokens=8, overlap_tokens=0, overflow=:paragraph)
    @test length(parts2) >= 2
    @test all(length(split(p)) <= 16 for p in parts2)

    # Hard token cap with overlap
    long = join(["w$i" for i in 1:50], " ")
    parts3 = split_to_token_budget(long, max_tokens=10, overlap_tokens=2, overflow=:token)
    @test length(parts3) >= 5
    # Each piece except the first should contain overlap tokens from previous
    @test length(split(parts3[2])) <= 12   # max_tokens + overlap
end
