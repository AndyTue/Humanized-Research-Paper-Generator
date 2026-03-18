import os
import re
from groq import Groq

# ── IEEE section headers used to split the paper ────────────────────────────
SECTION_PATTERNS = [
    r"^#+\s*(title)",
    r"^#+\s*(author)",
    r"^#+\s*(abstract)",
    r"^#+\s*(index terms)",
    r"^#+\s*\d+[\.\)]\s*(introduction)",
    r"^#+\s*\d+[\.\)]\s*(literature review|related work)",
    r"^#+\s*\d+[\.\)]\s*(methodology|methods)",
    r"^#+\s*\d+[\.\)]\s*(results)",
    r"^#+\s*\d+[\.\)]\s*(discussion)",
    r"^#+\s*\d+[\.\)]\s*(conclusion)",
    r"^#+\s*(acknowledgment|acknowledgement)",
    r"^#+\s*(references)",
]

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
    "abstract": (
        "Rewrite this Abstract so it reads like a human researcher summarising their own work. "
        "Use natural academic phrasing. Vary sentence length — some short, some longer. "
        "Avoid AI giveaways: no 'It is worth noting', 'Additionally', 'Furthermore' as openers. "
        "Keep every inline citation exactly as-is (e.g., [1], [2])."
    ),
    "introduction": (
        "Rewrite this Introduction as a human researcher would write it — with genuine curiosity "
        "and motivation. Open with a concrete observation or question, not a definition. "
        "Mix short punchy sentences with longer analytical ones (burstiness). "
        "Use first-person plural ('We', 'Our') where natural for the field. "
        "Preserve all inline citations exactly."
    ),
    "literature review": (
        "Rewrite this Literature Review section to sound like a researcher who has genuinely read "
        "these works and is offering their own synthesis. Use varied transitions: some contrasting "
        "('Yet', 'In contrast'), some building ('Building on this'), some temporal ('Earlier work by'). "
        "Avoid lists of author + year without commentary. Preserve all inline citations exactly."
    ),
    "related work": (
        "Same as literature review instructions: synthesise, contrast, and connect ideas naturally. "
        "Preserve all inline citations exactly."
    ),
    "methodology": (
        "Rewrite this Methodology to sound like a researcher explaining their own design decisions. "
        "Explain the 'why' behind choices, not just the 'what'. Use active voice where possible. "
        "Vary sentence structure — avoid starting every sentence the same way. "
        "Preserve all inline citations exactly."
    ),
    "methods": (
        "Same as methodology: active voice, explain rationale, vary structure. "
        "Preserve all inline citations exactly."
    ),
    "results": (
        "Rewrite this Results section to sound like a researcher reporting what they found — "
        "with a hint of genuine surprise or confirmation. Use precise language. "
        "Mix sentences that state a result with sentences that contextualise it. "
        "Preserve all inline citations and any numerical data exactly."
    ),
    "discussion": (
        "Rewrite this Discussion so it feels like a researcher thinking aloud about their findings. "
        "Include hedging language where appropriate ('This may suggest', 'One interpretation is'). "
        "Engage critically with prior work. Use conversational academic prose — not robotic recitation. "
        "Preserve all inline citations exactly."
    ),
    "conclusion": (
        "Rewrite this Conclusion to feel like a researcher reflecting genuinely on what was achieved. "
        "Avoid 'In conclusion' as the opening. Start with the most important takeaway. "
        "Be honest about limitations. End with a forward-looking but grounded statement. "
        "Preserve all inline citations exactly."
    ),
    "full_paper": (
        "Rewrite this complete research paper so it sounds like it was written by a real human academic. "
        "Apply these techniques throughout: (1) Vary sentence length deliberately — mix short impactful "
        "sentences with longer analytical ones (burstiness). (2) Avoid AI openers like 'It is important "
        "to note', 'Additionally', 'Moreover', 'Furthermore' at the start of paragraphs. "
        "(3) Use first-person plural ('We', 'Our') where natural. (4) Add hedging and nuance in the "
        "Discussion and Conclusion. (5) Use varied transition words. "
        "PRESERVE: all section headers, author name 'Andres TurrIzA', all inline citations [N], "
        "all numerical data, and the References list verbatim."
    ),
}

GENERIC_INSTRUCTION = (
    "Rewrite this academic section to sound like a human researcher wrote it. "
    "Vary sentence length (burstiness). Use natural academic voice. "
    "Avoid AI clichés. Preserve all inline citations exactly."
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
You are an expert human academic editor specialising in making AI-generated text indistinguishable from human-written academic prose.

Your rewriting technique:
- HIGH BURSTINESS: deliberately alternate between short sentences (5–10 words) and long analytical sentences (25–40 words).
- HIGH PERPLEXITY: choose less predictable word choices — prefer precise academic vocabulary over generic terms.
- REMOVE AI TELLS: never start a paragraph with "Additionally", "Furthermore", "Moreover", "It is important to note", "It is worth noting", "In conclusion".
- PRESERVE CITATIONS: every inline citation [1], [2], etc. MUST appear in the rewritten text in the same position.
- PRESERVE STRUCTURE: do not add or remove section headers. Do not change the author name.
- OUTPUT: return ONLY the rewritten body text, with no preamble or explanation.

SECTION-SPECIFIC INSTRUCTION:
{instruction}
"""

        user_prompt = f"SECTION TO REWRITE:\n\n{body.strip()}"

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                model=self.model,
                temperature=0.65,
                max_tokens=4000,
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