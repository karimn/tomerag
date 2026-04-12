using SHA

"""
    normalize_text(s) -> String

Lowercase, collapse whitespace. Used for dedup hashing (not for embedding input).
"""
function normalize_text(s::AbstractString)
    lowered = lowercase(s)
    stripped = strip(lowered)
    return replace(stripped, r"\s+" => " ")
end

"""
    token_count(s) -> Int

Whitespace-split token count. Fast approximation — adequate for chunk budgeting.
"""
token_count(s::AbstractString) = length(split(s))

"""
    content_hash(s) -> String

SHA-256 of the normalized text, lowercase hex. Used for dedup within a (source, doc).
"""
function content_hash(s::AbstractString)
    bytes = sha256(codeunits(normalize_text(s)))
    return bytes2hex(bytes)
end
