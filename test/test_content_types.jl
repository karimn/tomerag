using Test
using TomeRAG: DEFAULT_CONTENT_TYPES, PBTA_CONTENT_TYPES, YZE_CONTENT_TYPES

@testset "content_types" begin
    @test :mechanic in DEFAULT_CONTENT_TYPES
    @test :lore in DEFAULT_CONTENT_TYPES
    @test :procedure in DEFAULT_CONTENT_TYPES
    @test :boxed_text in DEFAULT_CONTENT_TYPES

    @test :move in PBTA_CONTENT_TYPES
    @test :playbook in PBTA_CONTENT_TYPES
    @test DEFAULT_CONTENT_TYPES ⊆ PBTA_CONTENT_TYPES

    @test :faction in YZE_CONTENT_TYPES
    @test :gear in YZE_CONTENT_TYPES
    @test DEFAULT_CONTENT_TYPES ⊆ YZE_CONTENT_TYPES
end
