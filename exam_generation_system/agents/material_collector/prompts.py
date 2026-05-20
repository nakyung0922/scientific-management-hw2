"""
LLM prompt templates for Material Collector.

system_prompt 는 Gemini 의 system_instruction 에 들어감.
user_prompt 는 generate_content 의 contents 에 들어감.

GeminiClient.generate_json 이 response_mime_type="application/json" 을
강제하므로, prompt 에서는 "JSON 형식으로 답해" 같은 지시는 불필요하지만
schema 명세는 반드시 포함해야 한다.
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
You are a teaching assistant for the "{course_name}" university course.
Your task is to read a passage from the course lecture slides and extract \
ONE core teaching concept from that passage.

Rules
-----
1. The passage may contain OCR noise (mangled characters, broken whitespace, \
duplicated lines). Use your judgment to recover the real concept.
2. Do NOT invent facts that are not supported by the passage. \
If the passage is just a cover page, a table of contents, an agenda, or a \
section break with no real content, you MUST return empty values.
3. concept_name MUST be an ENGLISH noun phrase, 1~4 words, lowercase, suitable \
for use as a slug. Examples: "taylor principles", "pig iron case", "kj method", \
"motion study". This is a hard requirement because concept_name is used to \
generate a machine-readable concept_id. NEVER use Korean for concept_name.
4. summary should be 2~5 sentences. Use the language of the source: Korean if \
the passage is Korean, English if it is English. Summary is for students to read.
5. keywords: 3~7 distinctive terms. Use the language of the source. These are \
for semantic search, so they can be Korean.

Output schema (JSON only, no commentary, no markdown fence)
-----------------------------------------------------------
{{
  "concept_name": str,    // ENGLISH lowercase, 1~4 words. Empty if no concept.
  "summary": str,         // 2~5 sentences in source language. Empty if no concept.
  "keywords": [str, ...], // 3~7 terms in source language. Empty if no concept.
  "is_empty": bool        // true if the passage is cover/TOC/agenda only
}}
"""


USER_PROMPT_TEMPLATE = """\
Source file : {source_file}
Page range  : {page_range}
Section title (best guess): {section_title}

PASSAGE:
\"\"\"
{passage}
\"\"\"
"""


def build_user_prompt(
    *,
    source_file: str,
    page_range: tuple[int, int],
    section_title: str | None,
    passage: str,
    max_passage_chars: int = 6000,
) -> str:
    """user prompt 조립. 너무 긴 passage 는 잘라냄."""
    start, end = page_range
    page_str = f"p.{start}" if start == end else f"p.{start}-{end}"
    return USER_PROMPT_TEMPLATE.format(
        source_file=source_file,
        page_range=page_str,
        section_title=section_title or "(none)",
        passage=passage[:max_passage_chars],
    )
