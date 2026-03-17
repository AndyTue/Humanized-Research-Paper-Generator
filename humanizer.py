import os
from groq import Groq

class Humanizer:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        
    def humanize_text(self, paper_text):
        """
        Rewrites a paper to sound like a natural human academic writer.
        Avoids robotic tone, repetitive sentence structures, and overly verbose AI tropes.
        Preserves the IEEE structure and meaning.
        """
        system_prompt = """
You are an expert human academic editor. Your job is to take a draft AI-generated research paper and humanize it.
The draft paper is written in plain text/markdown. You MUST NOT use LaTeX.
You MUST preserve the EXACT meaning, facts, numerical references like "[1]", "[2]", and the STRICT IEEE structure (all section headers must remain intact). The author MUST remain "Andres TurrIA".
Your goal is to rewrite the text so that it sounds like it was written by a natural human academic.

Key principles for humanization:
1. Vary sentence length and structure (avoid monotonous repetitive beginnings).
2. Remove common AI tropes (e.g., "In conclusion", "It is important to note", "Additionally", "Moreover" used excessively).
3. Soften the tone to be less robotic and more reflective of genuine academic inquiry.
4. Ensure transitions between paragraphs flow logically and naturally.
5. Do NOT change any of the formatting or section headers.
6. Make sure that the references remain explicitly numbered at the end of sentences, e.g., "... intelligence. [2]."
"""

        user_prompt = f"""
Here is the draft IEEE research paper. Please humanize it.

DRAFT PAPER:
{paper_text}
"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.6, # slightly lower for editing/humanizing
            )
            
            humanized_paper = response.choices[0].message.content
            return humanized_paper
        except Exception as e:
            print(f"Error during humanization: {e}")
            return paper_text # Fallback to original if humanization fails
