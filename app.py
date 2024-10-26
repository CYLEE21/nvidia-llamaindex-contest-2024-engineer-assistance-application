import getpass
import gradio as gr
import os
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = getpass.getpass("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")
    assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key

# Here we are using mixtral-8x7b-instruct-v0.1 model from API Catalog
Settings.llm = NVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400)

index = None
query_engine = None

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

def load_documents(file_objs):
    global index, query_engine
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
        
        # Simple preprocess the raw data
        
        index = VectorStoreIndex.from_documents(documents)
        #query_engine = index.as_query_engine(similarity_top_k=10)
        query_engine = index.as_query_engine(
            similarity_top_k=40, node_postprocessors=[NVIDIARerank(top_n=5)]
        )
        return f"successfully load {len(documents)} files!"
    
    except Exception as e:
        return f"Error: Error in load_documents. {e}"
    
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

def stream_process(message, history):
    global query_engine
    if query_engine is None:
        yield history + [(f"Please load documents first.", None)]
        return 
    
    try:
        response = query_engine.query(message)
        partial_response = ""
        for text in response.response_gen:
            partial_response += text
            yield history + [(message, partial_response)]
    except Exception as e:
        return history + [(message, f"Error processing query: {str(e)}")]

with gr.Blocks() as demo:
    gr.Markdown("# My first RAG application!")
    with gr.Row():
        file_input = gr.File(label="Select File to load.", file_count="multiple")
        load_btn = gr.Button("Load documents")

    load_output = gr.Textbox(label="Load status")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Please ask the questions", interactive=True)
    clear = gr.ClearButton([msg, chatbot])

    # event handler
    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
    msg.submit(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(lambda: "", outputs=[msg])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')


