async def call_llm(system: str, user: str, max_tokens: int = 500, tool: dict | None = None):
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if tool:
        kwargs["tools"] = [tool]
        kwargs["tool_choice"] = {"type": "tool", "name": tool["name"]}
    try:
        return await client.messages.create(**kwargs)
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")


def tool_output(resp) -> dict:
    block = next((b for b in resp.content if b.type == "tool_use"), None)
    if block is None:
        raise HTTPException(502, "Model did not return structured output")
    return block.input


def usage(resp) -> dict:
    return {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
