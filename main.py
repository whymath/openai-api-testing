
import os
import logging
import asyncio
from openai import OpenAI, AsyncOpenAI

from utils import query_api_sync, query_api_async

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sync_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    query_api_sync(sync_client, messages=[{"role": "system", "content": "Hello, how are you?"}])

    async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    asyncio.run(query_api_async(async_client, messages=[{"role": "system", "content": "Hello, how are you?"}]))
