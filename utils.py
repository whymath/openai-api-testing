
import logging
from openai import OpenAI, AsyncOpenAI


def query_api_sync(client: OpenAI, messages: list[dict[str, str]], model = "gpt-4o-mini", temperature: int = 0.1, max_tokens: int = 256) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages
    )
    output_message = response.choices[0].message.content
    logging.info(f"API response: {response}")
    logging.info(f"Output message: {output_message}")
    return output_message


async def query_api_async(client: AsyncOpenAI, messages: list[dict[str, str]], model = "gpt-4o-mini", temperature: int = 0.1, max_tokens: int = 256) -> str:
    response = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages
    )
    output_message = response.choices[0].message.content
    logging.info(f"API response: {response}")
    logging.info(f"Output message: {output_message}")
    return output_message
