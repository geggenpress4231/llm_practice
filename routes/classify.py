class ClassifyReq(BaseModel):
    text: str
    labels: list[str] = Field(min_length=2)

@app.post("/classify")
async def classify(req: ClassifyReq):
    tool = {
        "name": "record_classification",
        "description": "Record the classification of the text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": req.labels},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string", "description": "One sentence justification"},
            },
            "required": ["label", "confidence", "reason"],
        },
    }
    system = "Classify the text into exactly one of the allowed labels."
    resp = await call_llm(system, req.text, max_tokens=300, tool=tool)
    return {**tool_output(resp), "usage": usage(resp)}

