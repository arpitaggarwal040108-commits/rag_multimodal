
import re
from collections import Counter

from app.services.vision import explain_research_images


_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this",
    "from", "have", "has", "had", "you", "your", "their", "they", "them",
    "into", "about", "than", "then", "when", "where", "what", "which",
    "who", "how", "why", "can", "could", "would", "should", "will",
    "not", "but", "also", "more", "most", "some", "such", "each",
    "other", "using", "used", "use", "based", "between", "within",
    "paper", "study", "studies", "research", "results", "shown",
    "figure", "table", "section", "page", "chunk", "explain", "describe",
    "discuss", "tell", "give", "find", "according", "following",
    "compare", "comparison", "summarize", "summary", "provide",
}

MIN_SENTENCE_LENGTH = 30
MAX_EXTRACTED_FINDINGS = 10
MAX_FINDINGS_PER_CHUNK = 3
MAX_CONTEXT_CHARS = 7000


def _terms(text: str) -> set[str]:
    words = re.findall(
        r"[a-z0-9]+(?:[-_][a-z0-9]+)*",
        text.lower(),
    )
    return {
        word
        for word in words
        if len(word) > 2 and word not in _STOPWORDS
    }


def _clean_text(text: str) -> str:
    text = str(text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        _clean_text(text),
    )
    return [
        part.strip()
        for part in parts
        if len(part.strip()) >= MIN_SENTENCE_LENGTH
    ]


def _score_sentence(
    sentence: str,
    query_terms: set[str],
    document_frequency: Counter,
    total_chunks: int,
) -> float:
    sentence_terms = _terms(sentence)

    if not query_terms or not sentence_terms:
        return 0.0

    matched = query_terms & sentence_terms

    if not matched:
        return 0.0

    coverage = len(matched) / len(query_terms)

    idf = {
        term: 1.0
        + (
            (total_chunks + 1)
            / (document_frequency.get(term, 0) + 1)
        ) ** 0.5
        for term in query_terms
    }

    total_weight = sum(idf.values())
    matched_weight = sum(idf[term] for term in matched)

    weighted_coverage = (
        matched_weight / total_weight
        if total_weight
        else coverage
    )

    return 0.60 * coverage + 0.40 * weighted_coverage


def _source_name(chunk: dict) -> str:
    return str(
        chunk.get("original_filename")
        or chunk.get("source")
        or "unknown"
    )


def _chunk_identifier(chunk: dict):
    return chunk.get("chunk_id", chunk.get("global_id", "unknown"))


def _citation(chunk: dict) -> str:
    source = _source_name(chunk)
    chunk_id = _chunk_identifier(chunk)
    page = chunk.get("page")

    citation = f"[Source: {source}"

    if chunk_id != "unknown":
        citation += f", Chunk {chunk_id}"

    if page is not None:
        citation += f", Page {page}"

    return citation + "]"


def _source_record(chunk: dict) -> dict:
    chunk_id = _chunk_identifier(chunk)

    return {
        "source": chunk.get("source", _source_name(chunk)),
        "original_filename": _source_name(chunk),
        "chunk_id": chunk_id,
        "page": chunk.get("page"),
        "chunk_index": chunk.get("global_id", chunk_id),
        "score": chunk.get("score"),
        "text": chunk.get("text", ""),
    }


def _abstention(message: str) -> dict:
    return {
        "answer": message,
        "sources": [],
    }


def _prepare_text_context(chunks: list[dict]) -> str:
    """
    Prepare ranked text evidence for the vision model.

    Keep source and page metadata attached to each chunk. Do not modify
    stored chunks, embeddings, or indexes.
    """
    sections = []
    used_chars = 0

    for chunk in chunks:
        text = _clean_text(chunk.get("text", ""))

        if not text:
            continue

        chunk_id = _chunk_identifier(chunk)
        score = chunk.get("score")
        metadata = (
            f"[Source: {_source_name(chunk)}, "
            f"Page: {chunk.get('page', 'unknown')}"
        )
        if chunk_id != "unknown":
            metadata += f", Chunk: {chunk_id}"
        if score is not None:
            metadata += f", Retrieval score: {score}"
        section = metadata + "]\n" + text

        remaining = MAX_CONTEXT_CHARS - used_chars

        if remaining <= 0:
            break

        if len(section) > remaining:
            # Retain the available portion of the final section.
            section = section[:remaining]

        sections.append(section)
        used_chars += len(section) + 2

        if used_chars >= MAX_CONTEXT_CHARS:
            break

    return "\n\n".join(sections)


def _generate_text_answer(
    question: str,
    chunks: list[dict],
) -> dict:
    """
    Generate a structured, source-cited extractive answer.

    This fallback selects evidence sentences from retrieved chunks.
    It does not claim to be a complete literature-wide synthesis.
    """
    query_terms = _terms(question)

    if not query_terms:
        return _abstention(
            "Please enter a more specific research question."
        )

    document_frequency = Counter()

    for chunk in chunks:
        document_frequency.update(
            _terms(chunk.get("text", ""))
        )

    candidates = []
    seen_sentences = set()

    for chunk_position, chunk in enumerate(chunks):
        chunk_text = _clean_text(chunk.get("text", ""))

        for sentence in _split_sentences(chunk_text):
            normalized = re.sub(
                r"\W+",
                " ",
                sentence.lower(),
            ).strip()

            if not normalized or normalized in seen_sentences:
                continue

            seen_sentences.add(normalized)

            score = _score_sentence(
                sentence,
                query_terms,
                document_frequency,
                max(len(chunks), 1),
            )

            if score >= 0.16:
                candidates.append({
                    "score": score,
                    "sentence": sentence,
                    "chunk": chunk,
                    "chunk_position": chunk_position,
                })

    if not candidates:
        return _abstention(
            "The retrieved text does not contain enough matching evidence "
            "to answer this question reliably. Try a more specific question "
            "or check whether the relevant paper has been indexed."
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = []
    findings_per_chunk = Counter()
    selected_terms = set()

    for item in candidates:
        chunk_position = item["chunk_position"]

        if (
            findings_per_chunk[chunk_position]
            >= MAX_FINDINGS_PER_CHUNK
        ):
            continue

        sentence_terms = _terms(item["sentence"])

        if selected and sentence_terms:
            new_terms = sentence_terms - selected_terms
            novelty = len(new_terms) / len(sentence_terms)

            if novelty < 0.20:
                continue

        selected.append(item)
        findings_per_chunk[chunk_position] += 1
        selected_terms.update(sentence_terms)

        if len(selected) >= MAX_EXTRACTED_FINDINGS:
            break

    if not selected:
        return _abstention(
            "No sufficiently distinct evidence passages were found "
            "for this question."
        )

    sources = []
    seen_sources = set()
    finding_lines = []

    for index, item in enumerate(selected, start=1):
        chunk = item["chunk"]

        finding_lines.append(
            f"- **Finding {index}:** {item['sentence']} "
            f"{_citation(chunk)}"
        )

        source_record = _source_record(chunk)
        source_key = (
            source_record["source"],
            source_record["chunk_id"],
            source_record["page"],
        )

        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(source_record)

    answer = (
        "## Research evidence\n\n"
        "The following findings are extracted from the retrieved text. "
        "They are evidence passages rather than an independently verified "
        "literature-wide synthesis.\n\n"
        "## Key findings\n\n"
        + "\n\n".join(finding_lines)
        + "\n\n## Interpretation\n\n"
        "Read each finding alongside its cited source passage. The selection "
        "is based on overlap with your question and may not represent every "
        "contribution or conclusion in the papers.\n\n"
        "## Limitations\n\n"
        "This answer uses only the retrieved text passages. It does not "
        "independently verify claims or establish publication recency. "
        "Figures and diagrams require visual analysis when their contents "
        "are relevant."
    )

    return {
        "answer": answer,
        "sources": sources,
    }


def _normalize_visual_answer(answer: str) -> str:
    """Normalize model Markdown and remove common repeated output fragments."""
    answer = str(answer or "").replace("\r\n", "\n").replace("\r", "\n")
    answer = answer.replace("\x00", "")
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    answer = re.sub(
        r"(?im)^(#{1,3}\s*)irect answer\s*$",
        r"\1Direct answer",
        answer,
    )

    # Remove the known one-token output fragment emitted by some small local models.
    answer = re.sub(r"(?im)^\s*s\.\s*$\n?", "", answer)

    # Remove duplicate headings even when a blank line separates the heading and label.
    lines = answer.split("\n")
    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*#{1,3}\s+", line):
            heading_match = re.match(r"^\s*#{1,3}\s+(.+?)\s*#*\s*$", line)
            output.append(line)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if heading_match and j < len(lines):
                heading = heading_match.group(1).strip().rstrip(":").casefold()
                candidate = lines[j].strip()
                candidate_heading = re.match(r"^#{1,3}\s+(.+?)\s*#*\s*$", candidate)
                candidate_plain = re.sub(r"[*_`#]", "", candidate).strip().rstrip(":").casefold()
                same_heading = (
                    candidate_heading is not None
                    and candidate_heading.group(1).strip().rstrip(":").casefold() == heading
                ) or candidate_plain == heading
                if same_heading:
                    i = j + 1
                    continue
        else:
            output.append(line)
        i += 1

    text = "\n".join(output)
    table_lines = [
        line.strip() for line in text.splitlines()
        if "|" in line and not re.match(r"^\s*[-*+]\s+", line)
    ]
    has_markdown_table = (
        len(table_lines) >= 3
        and any(re.search(r"\|?\s*:?-{3,}:?\s*\|", line) for line in table_lines)
    )
    if has_markdown_table:
        text = "\n".join(
            line for line in text.splitlines()
            if not re.match(r"^\s*[-*+]\s+.*\|.*\|", line)
        )

    # Deduplicate repeated prose sentences across sections, but never rewrite
    # Markdown tables where identical placeholder cells can be legitimate.
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned_paragraphs = []
    seen_sentences = set()
    for paragraph in paragraphs:
        stripped_paragraph = paragraph.strip()
        if not stripped_paragraph:
            continue
        if "|" in stripped_paragraph and any(
            re.match(r"^\s*\|?.*\|.*$", line) for line in stripped_paragraph.splitlines()
        ):
            cleaned_paragraphs.append(stripped_paragraph)
            continue

        paragraph_lines = stripped_paragraph.splitlines()
        cleaned_lines = []
        for line in paragraph_lines:
            stripped_line = line.strip()
            is_heading = stripped_line.startswith("#")
            is_list_item = bool(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", stripped_line))
            if is_heading:
                cleaned_lines.append(line)
                continue

            prefix = ""
            body = stripped_line
            list_match = re.match(r"^([-*+]\s+|\d+[.)]\s+)(.*)$", stripped_line)
            if list_match:
                prefix, body = list_match.group(1), list_match.group(2)

            # Drop short lowercase continuation fragments that are detached
            # into their own paragraph by the local model (e.g. "and ...").
            if not is_list_item and len(body.split()) <= 12 and body[:1].islower():
                continue

            # Split first so a short detached continuation doesn't survive
            # merely because it shares a paragraph with a second sentence.
            sentences = re.split(r"(?<=[.!?])\s+", body)
            kept_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                # Small local models sometimes append lowercase tail fragments
                # as if they were new sentences. Drop these only when short and
                # detached; normal sentence-case prose and list items remain.
                if len(sentence.split()) <= 12 and sentence[:1].islower():
                    continue
                norm = re.sub(r"\W+", " ", sentence.casefold()).strip()
                if norm and norm in seen_sentences:
                    continue
                if norm:
                    seen_sentences.add(norm)
                kept_sentences.append(sentence)
            if kept_sentences:
                cleaned_lines.append(prefix + " ".join(kept_sentences))

        cleaned = "\n".join(cleaned_lines).strip()
        if cleaned:
            cleaned_paragraphs.append(cleaned)

    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(cleaned_paragraphs)).strip()


def _validate_visual_answer(answer: str) -> str:
    """
    Check whether the vision model returned the expected sections.

    This reports missing sections without fabricating findings or replacing
    the model's answer with invented evidence.
    """
    answer = _normalize_visual_answer(answer)

    required_headings = [
        "Direct answer",
        "Key findings",
        "Visual evidence",
        "Limitations",
    ]

    found_headings = {
        match.group(1).strip().lower()
        for match in re.finditer(
            r"^#{1,3}\s+(.+?)\s*$",
            answer,
            flags=re.MULTILINE,
        )
    }

    missing = [
        heading
        for heading in required_headings
        if heading.lower() not in found_headings
    ]

    if missing:
        answer += (
            "\n\n---\n\n"
            "**Response completeness warning:** "
            "The model omitted these expected sections: "
            + ", ".join(missing)
            + ". The answer may be incomplete; verify the cited evidence."
        )

    return answer



def _vision_question_with_evidence_rules(question: str) -> str:
    """Add task-specific evidence rules without changing retrieval or stored data."""
    q = question.casefold()
    rules = [
        "Answer the user's exact question. Do not substitute a related topic merely because it appears in the retrieved passages or images.",
        "Use only the supplied retrieved text and selected images as evidence.",
        "Treat retrieved passages as evidence candidates, not as proof that every sentence in them answers the question.",
        "Do not invent paper titles, authors, citations, page numbers, figure contents, method properties, experimental results, or numerical claims.",
        "Attach the exact source filename and printed page number to every material factual claim when metadata is available.",
        "If the retrieved evidence does not support a requested point, explicitly say that it is not established by the retrieved evidence.",
        "Keep source-reported facts, cautious inferences, and unknowns clearly separate.",
        "Do not repeat the same finding or limitation in multiple bullets.",
        "Do not claim to have measured accuracy, cost, latency, or scalability unless the supplied evidence reports the measurement and its evaluation setting.",
        "When an image is not legible or its content is not directly available, do not guess what it depicts; rely on the supplied text and say visual verification was unavailable.",
    ]
    comparison_terms = (
        "compare", "comparison", "trade-off", "tradeoff", "versus", " vs ",
        "difference between", "advantages and limitations",
    )
    metric_terms = (
        "accuracy", "computational cost", "latency", "scalability",
        "throughput", "memory", "runtime", "quantitative", "numerical",
        "performance",
    )
    visual_terms = ("figure", "chart", "plot", "diagram", "table", "image", "visual")

    if any(term in q for term in comparison_terms):
        rules.extend([
            "For comparisons, first identify the exact entities, methods, or dimensions named by the user. Cover each requested item explicitly and do not replace them with a neighboring comparison.",
            "If the question distinguishes two taxonomies or dimensions, explain that distinction before the table and do not conflate them.",
            "Include a Markdown comparison table. Preserve every column explicitly requested by the user; do not shorten or rename requested columns.",
            "For each row, use only evidence that specifically describes that row's method or dimension. Do not transfer a result from one method to another.",
            "If the user asks for accuracy, cost, latency, or scalability, provide reported values only when the retrieved text explicitly connects the value to the method, metric, dataset/evaluation setting, and source. Otherwise write exactly 'Not reported in the paper.'",
            "A benchmark table comparing named systems on datasets is not automatically a direct comparison of broad method categories. Explain this limitation when applicable.",
            "Do not describe different methods as identical unless the evidence explicitly supports that conclusion.",
        ])
    if any(term in q for term in metric_terms):
        rules.append(
            "Report metric values only when explicitly present in the supplied evidence. "
            "Otherwise write 'Not reported in the retrieved evidence'; never infer or fabricate a ranking."
        )
    if any(term in q for term in visual_terms):
        rules.extend([
            "Describe a figure, chart, or table only when its visible content or supplied text directly supports the description.",
            "Give a figure/table label and page only when visible or explicitly supplied.",
            "Distinguish what a visual literally shows from your interpretation.",
        ])

    rules.append(
        "Before drafting, silently map each requested sub-question and table column to specific supplied evidence. "
        "If a requested item has no supporting passage or legible visual evidence, mark it as not established instead of filling the gap."
    )
    return question.strip() + "\n\n[Answering and evidence rules]\n- " + "\n- ".join(rules)


def _deduplicate_source_records(chunks: list[dict]) -> list[dict]:
    """Return one source record per unique source/chunk/page combination."""
    records = []
    seen = set()
    for chunk in chunks:
        record = _source_record(chunk)
        key = (
            record.get("original_filename"),
            record.get("chunk_id"),
            record.get("page"),
        )
        if key not in seen:
            seen.add(key)
            records.append(record)
    return records


def generate_answer(
    question: str,
    chunks: list[dict],
    visual_results: list[dict] | None = None,
) -> dict:
    """
    Generate an answer using retrieved text and images when available.

    Preserves the existing function signature and response fields.
    """
    question = _clean_text(question)
    print("[InsightForge] generation grounding v3 active")

    if not question:
        return _abstention("Please enter a research question.")

    chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk, dict)
        and _clean_text(chunk.get("text", ""))
    ]

    visual_results = visual_results or []

    if not chunks and not visual_results:
        return _abstention(
            "No relevant text or visual evidence was retrieved."
        )

    text_context = _prepare_text_context(chunks)

    if visual_results:
        try:
            visual_answer = explain_research_images(
                question=question,
                visual_results=visual_results,
                text_context=text_context,
            )

            if visual_answer.get("available"):
                vision_text = _clean_text(visual_answer.get("answer", ""))

                # Do not return an empty vision response; use the safe text fallback.
                if not vision_text:
                    raise RuntimeError(
                        "The local vision model returned an empty answer."
                    )

                sources = _deduplicate_source_records(chunks)
                return {
                    "answer": _validate_visual_answer(vision_text),
                    "sources": sources,
                    "visual_sources": visual_answer.get(
                        "visual_sources", []
                    ),
                    "answer_mode": "local_vision",
                    "vision_model": visual_answer.get(
                        "model", "unknown"
                    ),
                }

        except RuntimeError as exc:
            fallback = _generate_text_answer(question, chunks)
            fallback["answer"] = (
                "## Visual analysis unavailable\n\n"
                f"{str(exc)}\n\n"
                "## Text evidence\n\n"
                + fallback["answer"]
            )
            fallback["answer_mode"] = "text_fallback"
            return fallback

    result = _generate_text_answer(question, chunks)
    result["answer_mode"] = "extractive_text"

    return result
