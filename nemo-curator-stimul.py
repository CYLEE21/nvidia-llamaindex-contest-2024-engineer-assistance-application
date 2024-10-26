import ftfy 
from nemo.collections.nlp.models import GPTModel
from openai import OpenAI


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/mixtral-8x7b-instruct-v0.1"  # Replace with your Hugging Face model name
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def text_reformatting(text):
    fixed_text = ftfy.fix_text(text)
    return fixed_text

def casual_analysis(text, model, api_key):
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = api_key
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role":"user",
            "content":f"""
                You are a customer complaint expert. Your task is to analyze provided documents to help development engineers identify issues based on your expertise and the information given.

                Please read the following documents carefully. They include causal relationships among them. For instance, the cause of certain customer complaints might be documented in the FMEA reports. However, keep in mind that since these documents are created by humans, there could be errors or unclear structures within the FMEA reports.

                Here's an overview of the documents for your reference:
                {text}.
            """
        }],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )
    return completion
    #for chunk in completion:
    #    if chunk.choices[0].delta.content is not None:
    #        yield print(chunk.choices[0].delta.content, end="")


ans = casual_analysis(model_name, "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")
print(ans)

for chunk in ans:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")