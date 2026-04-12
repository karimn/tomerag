using Test
using TomeRAG: HeuristicBackend, classify

@testset "heuristic classify" begin
    h = HeuristicBackend()

    # Move block: bold **When ...** trigger in text
    out = classify(h,
        text = "**When you delve the depths**, roll +wits. On a 10+...",
        heading_path = ["Moves", "Adventure Moves", "Delve the Depths"])
    @test out.content_type == :move
    @test out.move_trigger !== nothing
    @test occursin("delve the depths", lowercase(out.move_trigger))

    # Table
    out2 = classify(h,
        text = "| Roll | Result |\n|---|---|\n| 1 | A |\n| 2 | B |",
        heading_path = ["Reference", "Random Events"])
    @test out2.content_type == :table

    # Stat block
    out3 = classify(h,
        text = "NPC: Blight Walker\nHP 12, Armor 2\nAttack +3",
        heading_path = ["Bestiary", "Blight Walker"])
    @test out3.content_type == :stat_block
    @test out3.npc_name == "Blight Walker"

    # Fallback to lore via heading
    out4 = classify(h,
        text = "The corrupted forest stretches for miles...",
        heading_path = ["The World", "Geography"])
    @test out4.content_type == :lore
end
