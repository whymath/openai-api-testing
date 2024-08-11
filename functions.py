
import json
import logging
from duckduckgo_search import DDGS


def add_tool(tool_defn: dict, tools: list[dict] = None) -> list:
    if tools is None:
        tools = []
    tool = {
        "type": "function",
        "function": tool_defn,
    }
    tools.append(tool)
    return tools


###### Search DDG Functions Start ###### 


def search_ddg(search_query, max_results: int = 5):
    with DDGS() as search_client:
        search_results = search_client.text(search_query, max_results=max_results)
        search_results_str = "\n".join(result["body"] for result in search_results)
    logging.info(f"Search results: {search_results_str}")
    return search_results_str


def get_search_ddg_fn():
    search_ddg_function = {
        "name": "search_ddg",
        "description": "Search the internet using a search engine. Call this whenever you need to answer questions about topics that you're unsure about, including recent events; for example, when the user asks 'What happened in the 2024 Olympics on August 10. 2024?'",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "The query to search for on the internet.",
                },
            },
            "required": ["search_query"],
            "additionalProperties": False,
        }
    }
    return search_ddg_function


def generate_search_ddg_fn_result(tool_call, arguments, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    search_query = arguments.get('search_query')
    search_results = search_ddg(search_query)
    logging.info(f"Search results: {search_results}")

    function_call_result_message = {
        "role": "tool",
        "content": json.dumps({
            "search_query": search_query,
            "search_results": search_results
        }),
        "tool_call_id": tool_call.id
    }
    messages.append(function_call_result_message)
    return messages


###### Search DDG Functions End ######
