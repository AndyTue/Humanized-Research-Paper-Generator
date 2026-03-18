import os
from groq import Groq

REQUIRED_SECTIONS = [
    "abstract",
    "introduction",
    "literature review",
    "methodology",
    "results",
    "discussion",
    "conclusion",
    "references"
]

def count_words(text: str) -> int:
    return len(text.split())

def validate_ieee_structure(paper_text: str):
    """
    Validates IEEE sections AND minimum word count (2000 words).
    Returns (is_valid, missing_sections, word_count).
    """
    missing_sections = []
    text_lower = paper_text.lower()
    for section in REQUIRED_SECTIONS:
        if section not in text_lower:
            missing_sections.append(section)

    word_count = count_words(paper_text)
    is_valid = len(missing_sections) == 0 and word_count >= 2000
    return is_valid, missing_sections, word_count


class Generator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        self.min_words = 2000
        self.max_tokens = 8000  # Enough headroom for 2000+ word academic papers

    def _build_system_prompt(self) -> str:
        return """
You are a senior academic researcher and IEEE paper author. Your task is to write a COMPLETE, DETAILED research paper strictly following the IEEE structure below.

AUTHOR: Always set the author name to EXACTLY: Andres TurrIzA

OUTPUT FORMAT: Plain text / Markdown. No meta-labels like "Front Matter:" or "Main Body:".

MANDATORY IEEE SECTIONS (all must be present):
    Title
    Author (MUST BE "Andres TurrIzA")
    Abstract           — 200-250 words minimum
    Index Terms        — at least 10 terms
    1. Introduction    — 300-400 words minimum. Motivate the problem clearly.
    2. Literature Review — 350-450 words minimum. Discuss at least 4 prior works with citations.
    3. Methodology     — 350-450 words minimum. Explain approach, steps, tools.
    4. Results         — 300-400 words minimum. Present findings with data where possible.
    5. Discussion      — 300-400 words minimum. Interpret results, compare to prior work.
    6. Conclusion      — 200-250 words minimum. Summarize contributions and future work.
    Acknowledgment     — 50-100 words.
    References         — at least 6 entries, numbered [1]-[N].

CRITICAL RULES:
1. The TOTAL paper MUST be AT LEAST 2000 WORDS. Write fully developed paragraphs in every section.
2. Use ONLY the provided context (user documents + Wikipedia) as your factual basis. Do NOT hallucinate citations.
3. Cite inline using bracketed numbers: "... deep learning architectures. [3]."
4. Every claim must have at least one inline citation.
5. Each section must be written as full academic prose — not bullet points or outlines.
6. Do NOT use vague filler text. Every sentence must add real substance.
"""

    def _build_user_prompt(self, query: str, context: str, extra_note: str = "") -> str:
        base = f"""
Topic / Query: {query}

Provided Context:
{context}

IMPORTANT: Write the COMPLETE IEEE paper now. The paper MUST be at least 2000 words long.
Each section must be fully developed with multiple paragraphs of academic prose.
Do NOT produce a shortened or outline-style version — write the full paper.
"""
        if extra_note:
            base += f"\n\nNOTE FROM PREVIOUS ATTEMPT — YOU MUST FIX THIS: {extra_note}"
        return base

    def generate_ieee_paper(self, query: str, context: str, max_retries: int = 3) -> str:
        """
        Generates an IEEE paper. Validates structure AND word count.
        Retries up to max_retries if validation fails.
        """
        system_prompt = self._build_system_prompt()
        extra_note = ""

        for attempt in range(max_retries + 1):
            print(f"   Generation attempt {attempt + 1}/{max_retries + 1}...")
            try:
                user_prompt = self._build_user_prompt(query, context, extra_note)

                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    model=self.model,
                    temperature=0.7,
                    max_tokens=self.max_tokens,
                )

                paper = response.choices[0].message.content
                is_valid, missing_sections, word_count = validate_ieee_structure(paper)

                print(f"   -> Word count: {word_count} | Missing sections: {missing_sections}")

                if is_valid:
                    print(f"   OK Paper accepted ({word_count} words, all sections present).")
                    return paper

                # Build a specific correction note for the next retry
                issues = []
                if missing_sections:
                    issues.append(
                        f"Missing sections that MUST be added: {missing_sections}."
                    )
                if word_count < self.min_words:
                    deficit = self.min_words - word_count
                    issues.append(
                        f"Paper is only {word_count} words. You need at least {self.min_words}. "
                        f"Expand every section significantly — approximately {deficit // len(REQUIRED_SECTIONS)} "
                        f"more words per section on average."
                    )
                extra_note = " ".join(issues)

                if attempt < max_retries:
                    print(f"   FAIL Retrying... ({extra_note})")
                else:
                    print(f"   FAIL Max retries reached. Returning best paper ({word_count} words).")
                    return paper

            except Exception as e:
                print(f"   Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    return f"Failed to generate paper after {max_retries + 1} attempts. Last error: {e}"

        return "Failed to generate paper."