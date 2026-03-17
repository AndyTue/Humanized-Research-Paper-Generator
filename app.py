import gradio as gr
from agent import ResearchAgent

# Initialize the global agent
agent = ResearchAgent()

def process_uploaded_documents(files):
    if not files:
        return "Please upload at least one document (PDF, DOCX, TXT)."
    
    file_paths = [file.name for file in files]
    result = agent.process_files(file_paths)
    return result

def generate_paper(query):
    if not query.strip():
        return "Please enter a valid topic or query."
        
    if not agent.is_processed:
        return "Please upload and process documents first before generating a paper."
        
    result = agent.generate_research_paper(query)
    return result

with gr.Blocks(title="Humanized Research Paper Generator") as demo:
    gr.Markdown("# 📄 Humanized Research Paper Generator")
    gr.Markdown("Upload reference documents, input a topic, and generate a IEEE structured Research Paper, human-styled research paper powered by local vector search and Wikipedia.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Ingestion Phase")
            file_input = gr.File(label="Upload Documents (PDF, DOCX, TXT)", file_count="multiple", type="filepath")
            process_btn = gr.Button("Process Documents", variant="primary")
            process_output = gr.Textbox(label="Processing Status", interactive=False)
            
            gr.Markdown("### 2. Generation Phase")
            query_input = gr.Textbox(label="Topic / Query", placeholder="Text here your topic or query for create the Reseach Paper")
            generate_btn = gr.Button("Generate Research Paper", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### Generated Research Paper")
            paper_output = gr.Textbox(label="Output", lines=25, interactive=False)
            
    # Event wiring
    process_btn.click(
        fn=process_uploaded_documents,
        inputs=file_input,
        outputs=process_output
    )
    
    generate_btn.click(
        fn=generate_paper,
        inputs=query_input,
        outputs=paper_output
    )

if __name__ == "__main__":
    demo.launch()
