const DEFAULT_CONTENT_TYPES = Set{Symbol}([
    :mechanic, :lore, :adventure_scene, :table, :stat_block,
    :example, :gm_guidance, :flavor, :procedure, :boxed_text,
])

const PBTA_CONTENT_TYPES = DEFAULT_CONTENT_TYPES ∪ Set{Symbol}([
    :move, :gm_move, :playbook, :oracle, :front,
])

const YZE_CONTENT_TYPES = DEFAULT_CONTENT_TYPES ∪ Set{Symbol}([
    :oracle, :faction, :gear,
])
