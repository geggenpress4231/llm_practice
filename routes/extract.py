class ExtractReq(BaseModel):
    text: str
    fields: dict[str, str] = Field(
        description="field name -> description, e.g. {'invoice_number': 'The invoice ID'}"
    )

@app.post("/extract")
async def extract(req: ExtractReq):
    tool = {
        "name": "record_extraction",
        "description": "Record the extracted fields.",
        "input_schema": {
            "type": "object",
            "properties": {
                name: {"type": ["string", "null"], "description": desc}
                for name, desc in req.fields.items()
            },
            "required": list(req.fields),
        },
    }
    system = (
        "Extract the requested fields from the text. "
        "If a field is not explicitly present, return null. Never guess."
    )
    resp = await call_llm(system, req.text, max_tokens=600, tool=tool)
    return {"data": tool_output(resp), "usage": usage(resp)}