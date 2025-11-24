from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import httpx
from dotenv import load_dotenv
import base64
import imghdr

load_dotenv()

def _image_to_data_url(path: str) -> str:
    """Read a local image file and return a data URL (image/*;base64,...)"""
    with open(path, "rb") as f:
        raw = f.read()
    # try to determine image type (png, jpeg, gif, webp, etc.)
    kind = imghdr.what(None, raw) or "png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/{kind};base64,{b64}"

def llm_call(system_prompt: str, user_prompt: str | None = None, image_path: str | None = None):
    """
    Send a chat request to the model. Supports:
      - text only: provide user_prompt, leave image_path None
      - image only: provide image_path, leave user_prompt None
      - image + text: provide both

    The image is embedded as a markdown image using a data URL so the model
    receives the image inline with any text.
    """
    base_url = os.getenv("api_endpoint")
    api_key = os.getenv("api_key")
    model = os.getenv("model_gpt4o")

    client = httpx.Client(verify=False)

    llm = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        http_client=client
    )

    system_message = SystemMessage(content=system_prompt)

    # Build the user message content depending on inputs
    parts = []
    if image_path:
        try:
            data_url = _image_to_data_url(image_path)
            # add as markdown image so many chat models will see it
            parts.append(f"![user_image]({data_url})")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read/encode image: {e}")

    if user_prompt:
        parts.append(user_prompt)

    if not parts:
        raise ValueError("Provide at least user_prompt or image_path")

    user_content = "\n\n".join(parts)
    user_message = HumanMessage(content=user_content)

    # Invoke the model with system and user messages
    response = llm.invoke([system_message, user_message])

    print(response.content)
    return response.content

if __name__ == "__main__":
    # 1) Text only
    llm_call("You are a helpful assistant.", user_prompt="Explain special relativity simply.")

    # 2) Image only (path to local image)
    # llm_call("You are a helpful assistant.", image_path="sample-image.jpg")

    # 3) Image + text
    # llm_call(
    #     "You are a helpful assistant.",
    #     user_prompt="Summarize what is shown and list any visible errors.",
    #     image_path="sample-image.jpg"
    # )
