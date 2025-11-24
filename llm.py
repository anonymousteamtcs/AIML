from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import httpx
from dotenv import load_dotenv

load_dotenv()


def llm_call(system_prompt, user_prompt):
    base_url = os.getenv("api_endpoint")
    api_key = os.getenv("api_key")
    model = os.getenv("model_gpt4o")

    client = httpx.Client(verify=False)

    # Initialize the OpenAI chat model
    llm = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        http_client=client
    )

    # Define the system and user prompts
    system_prompt = SystemMessage(content=system_prompt)
    user_prompt = HumanMessage(content=user_prompt)

    # Call the LLM with both prompts
    response = llm.invoke([system_prompt, user_prompt])

    # Print the response
    print(response.content)

    return response.content

if __name__ == "__main__":
    system_prompt = "You are a helpful assistant."
    user_prompt = "Explain the theory of relativity in simple terms."
    llm_call(system_prompt, user_prompt)
