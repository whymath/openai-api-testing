
import os
import logging
import asyncio
from openai import OpenAI, AsyncOpenAI

from utils import query_api_sync, query_api_async, add_system_prompt, add_user_prompt, add_assistant_prompt, create_results_dir, log_messages_to_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    async_call = False

    messages = add_system_prompt(messages=None, prompt="You are a helpful assistant")
    messages = add_user_prompt(messages, prompt="Write a 50-word essay on OpenAI")

    if async_call:
        async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        output_message = asyncio.run(query_api_async(async_client, messages=messages))
    else:
        sync_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        output_message = query_api_sync(sync_client, messages=messages, max_tokens=512)
    
    messages = add_assistant_prompt(messages, prompt=output_message)
    results_dir = create_results_dir()
    log_messages_to_file(messages, results_dir, episode_name="episode_1")
    logging.info("All done!")
