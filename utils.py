
import logging
from datetime import datetime
from pathlib import Path
from openai import OpenAI, AsyncOpenAI


def add_system_prompt(messages: list[dict[str, str]], prompt: str) -> list[dict[str, str]]:
    if messages is None:
        messages = []
    messages.append({"role": "system", "content": prompt})
    return messages


def add_user_prompt(messages: list[dict[str, str]], prompt: str) -> list[dict[str, str]]:
    if messages is None:
        messages = []
    messages.append({"role": "user", "content": prompt})
    return messages


def add_assistant_prompt(messages: list[dict[str, str]], prompt: str) -> list[dict[str, str]]:
    if messages is None:
        messages = []
    messages.append({"role": "assistant", "content": prompt})
    return messages


def query_api_sync(client: OpenAI, messages: list[dict[str, str]], model = "gpt-4o-mini", temperature: int = 0.1, max_tokens: int = 256) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages
    )
    output_message = response.choices[0].message.content
    # logging.info(f"\nAPI response: {response}\n")
    logging.info(f"\nOutput message: {output_message}\n")
    return output_message


async def query_api_async(client: AsyncOpenAI, messages: list[dict[str, str]], model = "gpt-4o-mini", temperature: int = 0.1, max_tokens: int = 256) -> str:
    response = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages
    )
    output_message = response.choices[0].message.content
    # logging.info(f"\nAPI response: {response}\n")
    logging.info(f"\nOutput message: {output_message}\n")
    return output_message


def create_results_dir(exp_name: str = None) -> Path:
    dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if exp_name:
        dir_name += f"_{exp_name}"
    results_dir = Path("results", dir_name)
    results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir


def log_messages_to_file(messages: list[dict[str, str]], results_dir: Path, episode_name: str = None) -> None:
    logfile_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if episode_name:
        logfile_name += f"_{episode_name}"
    logfile_name += ".log"
    logfile_path = Path(results_dir, logfile_name)
    with open(logfile_path, "w") as f:
        for message in messages:
            # f.write("\n========================================================================\n")
            f.write(f"\n{message["role"].upper()} MESSAGE\n")
            f.write(f"\n{message["content"]}\n")
            f.write("\n========================================================================\n")
