using UUIDs

"""
    ingest!(registry, source_id, path; doc_id, document_type,
            format, embed_backend, classify_backend,
            extraction_backend) -> Int

Read `path`, chunk, classify, embed (batched), and insert into the source's
DuckDB store. Returns the number of chunks actually inserted (dedup-aware).

`format` defaults to `:auto` (inferred from file extension: `.pdf` → `:pdf`,
everything else → `:markdown`). Pass `:pdf` or `:markdown` explicitly to override.
`extraction_backend` is required when format resolves to `:pdf`.
"""
function ingest!(registry::SourceRegistry, source_id::AbstractString, path::AbstractString;
                 doc_id::AbstractString,
                 document_type::Symbol,
                 format::Symbol               = :auto,
                 embed_backend::EmbeddingBackend,
                 classify_backend::ClassifyBackend,
                 extraction_backend::Union{ExtractionBackend,Nothing} = nothing)

    # ── Resolve format ────────────────────────────────────────────────────────
    if format === :auto
        ext = lowercase(splitext(String(path))[2])
        format = ext == ".pdf" ? :pdf : :markdown
    end

    # Validate extraction_backend early (before touching registry/db).
    if format === :pdf && isnothing(extraction_backend)
        error("extraction_backend is required for format=:pdf")
    end

    src = get_source(registry, source_id)

    # ── Get document text ─────────────────────────────────────────────────────
    local doc_text::String
    if format === :pdf
        pages    = extract_pages(extraction_backend, String(path))
        # Inject page markers AFTER the first heading line so they land in the
        # section body (not the preamble, which the markdown parser discards).
        # For pages whose text has no heading, the marker is prepended instead
        # and will appear in the preamble of the NEXT section's flush buffer.
        function _inject_marker(page_text::String, page_num::Int)
            marker = "<!-- page $page_num -->"
            lines  = split(page_text, '\n'; limit=2)
            if length(lines) >= 1 && startswith(strip(lines[1]), "#")
                # Insert marker on the line immediately after the heading
                rest = length(lines) == 2 ? lines[2] : ""
                return "$(lines[1])\n$(marker)\n$(rest)"
            else
                return "$(marker)\n$(page_text)"
            end
        end
        doc_text = join(
            [_inject_marker(p.text, p.page_num) for p in pages],
            "\n\n",
        )
    elseif format === :markdown
        doc_text = read(path, String)
    else
        error("unsupported format: $format (expected :pdf or :markdown)")
    end

    # ── Chunk ─────────────────────────────────────────────────────────────────
    raws = chunk_document(doc_text, src.chunking)
    isempty(raws) && return 0

    # For PDF: extract page numbers from markers, strip markers from text.
    if format === :pdf
        raws = _assign_pages(raws)
    end

    # ── Classify ──────────────────────────────────────────────────────────────
    classified = [classify(classify_backend; text=r.text, heading_path=r.heading_path)
                  for r in raws]

    # ── Embed (batched) ───────────────────────────────────────────────────────
    embeddings = embed(embed_backend, [r.text for r in raws])

    # ── Build Chunk objects ───────────────────────────────────────────────────
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

"""
    _assign_pages(chunks) -> Vector{RawChunk}

Scans each chunk's text for `<!-- page N -->` markers injected during PDF extraction.
Sets `chunk.page` to the last marker found in that chunk's text (the page where the
body content lives). Strips all markers from the returned chunk text.
"""
function _assign_pages(chunks::Vector{RawChunk})
    map(chunks) do rc
        matches = collect(eachmatch(r"<!-- page (\d+) -->", rc.text))
        page    = isempty(matches) ? "" : String(matches[end][1])
        clean   = strip(replace(rc.text, r"<!-- page \d+ -->\n?" => ""))
        RawChunk(
            heading_path = rc.heading_path,
            text         = clean,
            chunk_order  = rc.chunk_order,
            page         = page,
        )
    end
end

_backend_model_name(::MockEmbeddingBackend) = "mock"
_backend_model_name(b::OllamaBackend)       = b.model
_backend_model_name(::EmbeddingBackend)     = ""
