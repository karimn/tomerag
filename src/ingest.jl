using UUIDs

"""
    ingest!(registry, source_id, path; doc_id, document_type, format=:markdown,
            embed_backend, classify_backend) -> Int

Read `path`, chunk, classify, embed (batched), and insert into the source's
DuckDB store. Returns the number of chunks actually inserted (dedup-aware).
Only `:markdown` format is supported in Plan 1.
"""
function ingest!(registry::SourceRegistry, source_id::AbstractString, path::AbstractString;
                 doc_id::AbstractString,
                 document_type::Symbol,
                 format::Symbol = :markdown,
                 embed_backend::EmbeddingBackend,
                 classify_backend::ClassifyBackend)
    src = get_source(registry, source_id)
    format == :markdown || error("Plan 1 supports only format=:markdown; got $format")

    text = read(path, String)
    raws = chunk_document(text, src.chunking)
    isempty(raws) && return 0

    # Classify all chunks.
    classified = [classify(classify_backend; text=r.text, heading_path=r.heading_path)
                  for r in raws]

    # Batch embed.
    embeddings = embed(embed_backend, [r.text for r in raws])

    chunks = Chunk[]
    for (i, r) in enumerate(raws)
        cls = classified[i]
        push!(chunks, Chunk(
            id              = string(uuid4()),
            source_id       = src.id,
            doc_id          = String(doc_id),
            doc_path        = abspath(path),
            text            = r.text,
            embedding       = embeddings[i],
            embedding_model = src.embedding_model,
            token_count     = token_count(r.text),
            content_hash    = content_hash(r.text),
            document_type   = document_type,
            system          = src.system,
            edition         = "",
            page            = r.page,
            heading_path    = r.heading_path,
            chunk_order     = r.chunk_order,
            parent_id       = nothing,
            content_type    = cls.content_type,
            tags            = cls.tags,
            move_trigger    = cls.move_trigger,
            scene_type      = cls.scene_type,
            encounter_key   = cls.encounter_key,
            npc_name        = cls.npc_name,
            license         = src.license,
        ))
    end
    return insert_chunks(src, chunks)
end

_backend_model_name(::MockEmbeddingBackend) = "mock"
_backend_model_name(b::OllamaBackend) = b.model
_backend_model_name(::EmbeddingBackend) = ""
