using Test
using TomeRAG: token_count, content_hash, normalize_text

@testset "tokenize" begin
    @test token_count("hello world") == 2
    @test token_count("  hello   world \n foo ") == 3
    @test token_count("") == 0

    @test normalize_text(" Hello\tWorld \n") == "hello world"

    h1 = content_hash("Hello World")
    h2 = content_hash("hello world")
    h3 = content_hash("hello  world")
    @test h1 == h2 == h3        # normalized before hashing
    @test length(h1) == 64      # sha256 hex
end
