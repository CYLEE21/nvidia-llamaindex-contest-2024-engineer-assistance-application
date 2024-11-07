import os
from dotenv import load_dotenv

import gradio as gr
from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from openai import OpenAI

from nemo_curator_dummy import nemo_curator_text_reformatting

load_dotenv()
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")

llm_model = "mistralai/mixtral-8x7b-instruct-v0.1"
embed_model = "NV-Embed-QA"

# Here we are using mixtral-8x7b-instruct-v0.1 model from API Catalog
Settings.llm = NVIDIA(model=llm_model)
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400)

index = None
query_engine = None
analysis_result = None

def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]


def load_documents(file_objs):
    global query_engine
    try:
        if not file_objs:
            return f"Error: No file selected."
        file_paths = get_files_from_input(file_objs)
        documents = []

        for file_path in file_paths:
            directory = os.path.dirname(file_path)
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

        if not documents:
            return f"Error: No documents found in the selected files."
    
    except Exception as e:
        print(f"Exception in loading the error {e}")
    
    try:
        global client
        client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = nvidia_api_key
        )
        # Using Nemo Curator
        documents = nemo_curator_text_reformatting(documents)
        
        response_analysis, history = init_casual_analysis(documents, history=[])

        documents.append(Document(text=response_analysis))
        
        query_engine = VectorStoreIndex.from_documents(documents).as_query_engine(
            similarity_top_k=40, node_postprocessors=[NVIDIARerank(top_n=5)]
        )

        initial_message = {"role": "assistant", "content": str(response_analysis)}
        
        history.append({"role": "assistant", "content": str(response_analysis)})
        
        return f"Successfully load {len(documents)} files! The summary has been transferred in the documents.", [initial_message]
    
    except Exception as e:
        return f"Error: error in using the model. {e}"


def init_casual_analysis(message, history):
    
    try:
        global llm_model
        global client
        
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{
                "role":"user",
                "content":f"""
                    You are a customer complaint expert. Your task is to analyze provided documents to help development engineers identify issues based on your expertise and the information given.

                    The documents included 1. FMEA reported provided by the engineers, 2. Fridge temperature log, 3. Developer manual, and 4. User manual.

                    Please read the following documents carefully. They include causal relationships among them. For instance, the cause of certain customer complaints might be documented in the FMEA reports. 
                    
                    However, keep in mind that since these documents are created by humans, there could be errors or unclear structures within the FMEA reports.

                    Here's an overview of the documents for your reference, please sumarize the documents's casuality and provide the answer as short and concise as possilbe within 500 token:

                    {message}.
                """
            }],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True
        )
        # Extract response content
        response_text = []
    
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                response_text.append(chunk.choices[0].delta.content)
        
        complete_string = "".join(response_text)
        
        return complete_string, history
    
    except Exception as e:
        return [(message, f"Error casuality analysis: {str(e)}")]
    

def chat(message, history):
    global query_engine
    
    if query_engine is None:
        return history + [(f"Please load documents first.", None)]
    
    try:
        response = query_engine.query(message)
        
        history.append({"role": "user", "content": str(message)})
        history.append({"role": "assistant", "content": str(response)})
        return "", history
    
    except Exception as e:
        return [(message, f"Error processing query: {str(e)}")]


with gr.Blocks() as demo:
    gr.Markdown("# My first RAG application!")
    with gr.Row():
        file_input = gr.File(label="Select File to load.", file_count="multiple")
        load_btn = gr.Button("Load documents")
    
    load_output = gr.Textbox(label="Load status")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Please ask the questions", interactive=True)
    clear = gr.ClearButton([msg, chatbot])

    # Event handler
    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output, chatbot])
    msg.submit(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(lambda: "", outputs=[msg])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')


