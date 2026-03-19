import os
import re
from groq import Groq

# Sections that should NOT be paraphrased — preserve verbatim
SKIP_SECTIONS = {"title", "author", "index terms", "references", "acknowledgment", "acknowledgement"}


def _identify_section_label(header_line: str) -> str:
    """Returns a normalised section label from a markdown header line."""
    clean = re.sub(r"^#+\s*", "", header_line).strip().lower()
    clean = re.sub(r"^\d+[\.\)]\s*", "", clean).strip()
    return clean


def split_paper_into_sections(paper_text: str) -> list[dict]:
    """
    Splits the paper into a list of {"header": str, "body": str, "label": str} dicts.
    Sections that share a header but have no dedicated body stay attached to their header.
    """
    lines = paper_text.splitlines(keepends=True)
    sections: list[dict] = []
    current_header = ""
    current_label = ""
    current_body_lines: list[str] = []

    header_re = re.compile(r"^#+\s+", re.MULTILINE)

    for line in lines:
        if header_re.match(line):
            # Save previous section
            if current_header or current_body_lines:
                sections.append({
                    "header": current_header,
                    "body": "".join(current_body_lines),
                    "label": current_label,
                })
            current_header = line.rstrip("\n")
            current_label = _identify_section_label(current_header)
            current_body_lines = []
        else:
            current_body_lines.append(line)

    # Flush last section
    if current_header or current_body_lines:
        sections.append({
            "header": current_header,
            "body": "".join(current_body_lines),
            "label": current_label,
        })

    # Handle papers that use no markdown headers at all (plain text IEEE)
    if len(sections) <= 1:
        return [{"header": "", "body": paper_text, "label": "full_paper"}]

    return sections


# ── Per-section humanization prompts ────────────────────────────────────────

SECTION_INSTRUCTIONS = {
    "abstract": "Rewrite abstract like a human researcher. Use natural phrasing, varying sentence length. No AI giveaways. Preserve inline citations [N].",
    "introduction": "Rewrite introduction with genuine curiosity. Mix sentence lengths. Use 'We'/'Our' if natural. Preserve citations.",
    "literature review": "Synthesise prior works naturally. Use varied transitions ('Yet', 'Building on this'). Preserve citations.",
    "related work": "Synthesise prior works naturally. Use varied transitions. Preserve citations.",
    "methodology": "Explain the 'why' behind choices, not just 'what'. Use active voice, varied structure. Preserve citations.",
    "methods": "Explain rationale, use active voice and varied structure. Preserve citations.",
    "results": "Report findings precisely. Contextualise results. Preserve citations and numerical data exactly.",
    "discussion": "Use hedging language ('This may suggest'). Engage critically. Conversational academic prose. Preserve citations.",
    "conclusion": "Reflect on achievements. Avoid 'In conclusion'. State main takeaway and limitations. Preserve citations.",
    "full_paper": "Rewrite paper naturally. Vary sentence length. Keep citations, data, and section headers exactly.",
}

GENERIC_INSTRUCTION = (
    "Rewrite to sound human. Vary sentence length. Preserve all citations exactly."
)


def _get_instruction(label: str) -> str:
    for key, instruction in SECTION_INSTRUCTIONS.items():
        if key in label:
            return instruction
    return GENERIC_INSTRUCTION


# ── Main Humanizer class ─────────────────────────────────────────────────────

class Humanizer:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    def _humanize_section(self, header: str, body: str, label: str) -> str:
        """
        Sends a single section to the LLM for humanization.
        Returns the rewritten body (header is preserved as-is).
        """
        instruction = _get_instruction(label)

        system_prompt = f"""
You are an expert human academic editor formatting text to be indistinguishable from human-written prose.
Rules:
1. High burstiness (mix short 5-10 word and long 25-40 word sentences).
2. Use precise, less predictable academic vocabulary.
3. NEVER start paragraphs with "Additionally", "Furthermore", "In conclusion".
4. PRESERVE ALL inline citations (e.g., [1]) EXACTLY as they appear.
5. NO preamble. RETURN ONLY rewritten text.

Section Instructions: {instruction}
"""

        user_prompt = f"Rewrite this:\n\n{body.strip()}"

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                model=self.model,
                temperature=0.65,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"   Warning: could not humanize section '{label}': {e}")
            return body  # Fall back to original section body

    def humanize_text(self, paper_text: str) -> str:
        """
        Multi-pass humanization pipeline:
          1. Split paper into IEEE sections.
          2. Humanize each non-skip section independently.
          3. Reassemble the paper.
        """
        print("   Splitting paper into sections...")
        sections = split_paper_into_sections(paper_text)
        print(f"   Found {len(sections)} section(s).")

        # If the paper couldn't be split, fall back to single full-paper call
        if len(sections) == 1 and sections[0]["label"] == "full_paper":
            print("   No section headers detected — humanizing full paper in one pass...")
            humanized_body = self._humanize_section("", paper_text, "full_paper")
            return humanized_body

        # Multi-section processing
        rebuilt_parts: list[str] = []

        for i, section in enumerate(sections):
            label = section["label"]
            header = section["header"]
            body = section["body"]

            # Always keep the header line
            if header:
                rebuilt_parts.append(header)

            # Skip sections that should not be rewritten
            if any(skip in label for skip in SKIP_SECTIONS):
                print(f"   Preserving section verbatim: '{label}'")
                if body.strip():
                    rebuilt_parts.append(body.rstrip())
                continue

            # Skip sections with very little content (likely just a label line)
            if len(body.strip()) < 80:
                print(f"   Skipping thin section (too short to rewrite): '{label}'")
                rebuilt_parts.append(body.rstrip())
                continue

            print(f"   Humanizing section {i + 1}/{len(sections)}: '{label}' ({len(body.split())} words)...")
            humanized_body = self._humanize_section(header, body, label)
            rebuilt_parts.append("\n" + humanized_body + "\n")

        result = "\n".join(rebuilt_parts)
        print("   Humanization complete.")
        return result