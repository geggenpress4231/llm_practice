import os, json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import HTTPException

load_dotenv()

client = AsyncOpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),
)
MODEL = os.getenv("MODEL", "llama3.2")


async def call_llm(system: str, user: str, max_tokens: int = 500, tool: dict | None = None):
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if tool:
        kwargs["tools"] = [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }]
        kwargs["tool_choice"] = {"type": "function", "function": {"name": tool["name"]}}
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")


def text_output(resp) -> str:
    return (resp.choices[0].message.content or "").strip()


def tool_output(resp) -> dict:
    calls = resp.choices[0].message.tool_calls
    if not calls:
        raise HTTPException(502, "Model did not return structured output")
    args = calls[0].function.arguments
    try:
        return json.loads(args) if isinstance(args, str) else args
    except json.JSONDecodeError:
        raise HTTPException(502, "Model returned malformed JSON arguments")


def usage(resp) -> dict:
    u = resp.usage
    return {
        "input_tokens": u.prompt_tokens if u else None,
        "output_tokens": u.completion_tokens if u else None,
    }