import os
from groq import Groq

def validate_ieee_structure(paper_text):
    """
    Validates if the generated paper contains the mandatory IEEE sections.
    """
    required_sections = [
        "abstract",
        "introduction",
        "literature review",
        "methodology",
        "results",
        "discussion",
        "conclusion",
        "references"
    ]
    
    missing_sections = []
    text_lower = paper_text.lower()
    for section in required_sections:
        if section.lower() not in text_lower:
            missing_sections.append(section)
            
    return len(missing_sections) == 0, missing_sections

class Generator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        
    def generate_ieee_paper(self, query, context, max_retries=2):
        """
        Generates an IEEE paper based on the query and context.
        Validates structure and retries if missing sections.
        """
        system_prompt = """
You are an expert academic researcher and writer. Your task is to generate a comprehensive research paper following the STRICT IEEE structure.
You MUST output the paper in plain text/markdown formatting. 
Do NOT include structural labels like "Front Matter:", "Main Body:", or "Back Matter:" in your output. Just output the sections directly with their headers.
You MUST set the author name explicitly to EXACTLY: Andres TurrIzA

You MUST use ONLY the provided context to write the paper. Do NOT make up facts or hallucinate citations.
If the context is insufficient, state it, but you MUST still include all sections.

STRICT IEEE STRUCTURE SECTIONS REQUIRED:
- Title
- Author (MUST BE "Andres TurrIzA")
- Abstract
- Index Terms
- 1. Introduction
- 2. Literature Review
- 3. Methodology
- 4. Results
- 5. Discussion
- 6. Conclusion
- Acknowledgment
- References

IMPORTANT NUMERICAL CITATIONS RULE:
At the end of each referenced sentence in the paper, you MUST explicitly show the reference number in brackets.
Example: "As research in this field continues to evolve, it is likely that deep learning will play an increasingly vital role in shaping the future of artificial intelligence. [2]."
Make sure your References section entries correspond directly to these numbers.
"""
        user_prompt = f"""
Topic/Query: {query}

Provided Context:
{context}

Please generate the full IEEE research paper now.
"""

        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model,
                    temperature=0.7,
                )
                
                generated_paper = response.choices[0].message.content
                
                is_valid, missing_sections = validate_ieee_structure(generated_paper)
                if is_valid:
                    return generated_paper
                else:
                    if attempt < max_retries:
                        print(f"Warning: Missing sections {missing_sections}. Retrying generation...")
                        user_prompt += f"\n\nIMPORTANT: Previous attempt missed the following sections: {missing_sections}. You MUST include them."
                    else:
                        print("Warning: Max retries reached. Returning paper with missing sections.")
                        return generated_paper
            except Exception as e:
                print(f"Error during generation: {e}")
                if attempt == max_retries:
                    return f"Failed to generate paper due to error: {e}"
                    
        return "Failed to generate paper."
