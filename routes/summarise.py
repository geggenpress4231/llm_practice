class SummarizeReq(BaseModel):
    text: str = Field(min_length=20, max_length=50_000)
    max_sentences: int = Field(3, ge=1, le=10)

@app.post("/summarize")
async def summarize(req: SummarizeReq):
    system = (
        "You are a precise summarizer. Use only information in the text. "
        f"Reply with at most {req.max_sentences} sentences and nothing else."
    )
    resp = await call_llm(system, req.text, max_tokens=400)
    return {"summary": resp.content[0].text.strip(), "usage": usage(resp)}