from langchain_openai import ChatOpenAI
import httpx
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# LLM setup (IMPORTANT)
# Please update the base_url, model, api_key as specified below.
base_url = os.getenv("api_endpoint")
api_key = os.getenv("api_key")

input_prompt = "hi testing model response"
client = httpx.Client(verify=False)

# Load model names from file
model_file = "models.txt"
with open(model_file, "r") as f:
    model_names = [line.strip() for line in f if line.strip()]

# Loop through models
for model_name in model_names:
    print(f"Testing model: {model_name}")
    
    # Create the LLM
    llm = ChatOpenAI(
        base_url=base_url,
        model=model_name,
        api_key=api_key,
        http_client=client
    )

    response = llm.invoke(input_prompt)
    print("Model Response:", response)
