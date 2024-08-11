
import os
import logging
import asyncio
from openai import OpenAI, AsyncOpenAI

from oat.functions import add_tool, get_search_ddg_fn, handle_tool_calls
from oat.utils import query_completions_api_sync, query_completions_api_async, add_system_prompt, add_user_prompt, add_assistant_prompt, create_results_dir, log_messages_to_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    async_call = False

    # Prepare the messages
    messages = add_system_prompt(messages=None, prompt="You are a helpful assistant")
    # messages = add_user_prompt(messages, prompt="Write a 50-word essay on Paris")
    messages = add_user_prompt(messages, prompt="What happened in the 2024 Olympics on August 10, 2024?")

    # Test searching on DuckDuckGo
    # search_results = search_ddg("What is the capital of France?")
    # logging.info(f"Test search results: {search_results}")

    # Create a tool for the search function
    tools = add_tool(get_search_ddg_fn())

    # Call the OpenAI API
    if async_call:
        async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        output_message, response = asyncio.run(query_completions_api_async(async_client, messages=messages, tools=tools, max_tokens=512))
    else:
        sync_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        output_message, response = query_completions_api_sync(sync_client, messages=messages, tools=tools, max_tokens=512)

    # Check if tool needs to be called
    output_message, response, messages = handle_tool_calls(sync_client, response, messages, tools)    

    # Save the results to a file
    messages = add_assistant_prompt(messages, prompt=output_message)
    results_dir = create_results_dir()
    log_messages_to_file(messages, results_dir, episode_name="episode_1")
    logging.info("All done!")
