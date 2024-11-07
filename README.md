# NVIDIA-Llamaindex Contest (Engineer Assistance Application)
Author: Chi-Yuan Lee

## 1 Abstract

This LLM project aims to participate Nvidia Llamainde contest 2024. In this project, I harnessed Nvidia NIM, Llamaindex, and OpenAI's technologies to construct a prototype to assist the users to diagnose the potential root cause based on the given documents. The project results indicate the feasibility of using LLM to assist the user to extract the information from the complex documentary. The repository includes the nescesary installments and data for the test. However, the user should adopt private api key.

## 2 Motivation

In production, the customer feedback, FMEA, and documentations are considered as the vital materials to help the company to improve the product and solve the issues. The relationship among the events might include causality and correlation. However, in practice, the issue analysis usually depands on the experience employee to track the root cause, which can be a challenge to team without the experience employees. 

This prototype provide a concept using LLM to assist the user to diagnoise the given issues and adviced the potential solutions.

## 3 Quick Start

### Step 1: clone this repo.

### Step 2: create an `.env` to include your `NVIDIA_API_KEY` as follows:

```
NVIDIA_API_KEY=nvapi-my_api
```

### Step 3: create a venv with python==3.9 and install requirements.

``` bash
python -m venv myenv

source venv/bin/activate

pip install -r requirements.txt
```

### Step 4: run the app.

``` bash
python app.py
```

### Step 5: upload the example data from repo folders and start the conversation.

## 4 Case study

In the project, I adopted a case study to demo the app. In the case study, a drink machine company is managing the product improvement based on the customer feedback. The drink machine can store the preprocessed fresh fruite packages in the build-in fridge and make the fresh juice when the customer order it. However, the process is complex and include the causality. Thus, based on the customer feedback, this app might be able to provide the potential root causes and solutions.

## 5 Results

### Question:

> Today morning, there is a customer feedback that the juice is not fresh, what can be the potential root cause? and how was the situation of today's fridge?

### Answer:

> Based on the context, the potential root cause for the juice not being fresh could be contaminants in the juicer or improper storage of juice ingredients. The fridge temperature for today, as per the log, has fluctuated throughout the day. There were instances where the temperature was quite high, which could affect the quality of the juice. However, without specific temperature readings, it is difficult to definitively say if the temperature was the cause of the juice not being fresh. It is recommended to check the juicer for any contaminants and ensure proper storage of juice ingredients. Additionally, the fridge temperature should be monitored to ensure it remains at a suitable level for storing juice ingredients.

## 6 Architecture

Raw data -> Dummy Nvidia Nemo Curator (simple data cleaning) -> Nividia NIM Microservices (preprocessing and summarization) -> Vector database (local memory) -> Nividia NIM Microservices (Reranking and conversation) -> User

Used models:

1. llm_model = "mistralai/mixtral-8x7b-instruct-v0.1"
2. embed_model = "NV-Embed-QA"

## 7 Challenges and future development

Challenges:
1. Real world practical scenario.
2. Libraries do not support to some specific environment (ex. MacOS).

Future Development:
1. Integrate more Nvidia eco-system libraries.
2. Fine-tune the LLM for more accurate and specific response.
3. Better UI design.

