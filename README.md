# Project Title: NVIDIA-Llamaindex Contest (Engineer Assistance Application)

## Abstract

This LLM project aims to participate Nvidia Llamainde contest 2024. In this project, I harnessed Nvidia NIM, Llamaindex, and OpenAI's technologies to construct a prototype to assist the users to diagnose the potential root cause based on the given documents. The project results indicate the feasibility of using LLM to assist the user to extract the information from the complex documentary. The repository includes the nescesary installments and data for the test. However, the user should adopt private api key.

## Quick Start

### Step 1: clone this repo.

### Step 2: create an `.env` to include your `NVIDIA_API_KEY` as follows:

```
NVIDIA_API_KEY=nvapi-my_api
```

### Step 3: create a venv with python 3.10 and install requirements.

``` bash
pip install -r requirements.txt
```

### Step 4: run the app.

``` bash
python app.py
```

## Results

### Question:
> Today morning, there is a customer feedback that the juice is not fresh, what can be the potential root cause? and how was the situation of today's fridge?

### Answer:
> Based on the context, the potential root cause for the juice not being fresh could be contaminants in the juicer or improper storage of juice ingredients. The fridge temperature for today, as per the log, has fluctuated throughout the day. There were instances where the temperature was quite high, which could affect the quality of the juice. However, without specific temperature readings, it is difficult to definitively say if the temperature was the cause of the juice not being fresh. It is recommended to check the juicer for any contaminants and ensure proper storage of juice ingredients. Additionally, the fridge temperature should be monitored to ensure it remains at a suitable level for storing juice ingredients.
