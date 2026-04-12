using Test
using TomeRAG

@testset "TomeRAG.jl" begin
    include("test_content_types.jl")
    include("test_types.jl")
    include("test_tokenize.jl")
    include("test_backends.jl")
    include("test_ollama.jl")
    include("test_heuristic.jl")
    include("test_chunker.jl")
    include("test_splitter.jl")
    include("test_storage.jl")
    include("test_ingest.jl")
end
