from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from helper.call_llm import call_llm, tool_output, usage

router = APIRouter()

class ClassifyReq(BaseModel):
    text: str
    labels: list[str] = Field(min_length=2)

@router.post("/classify")
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
    allowed = ", ".join(f'"{l}"' for l in req.labels)
    system = (
        "Classify the text into exactly one of the allowed labels.\n"
        f"Allowed labels: {allowed}.\n"
        "The label must be copied exactly from this list. Never invent a new label. "
        "If nothing fits well, choose the closest one, or 'other' if it is in the list."
    )

    user = req.text
    total = {"input_tokens": 0, "output_tokens": 0}
    attempts = 2

    for attempt in range(attempts):
        resp = await call_llm(system, user, max_tokens=300, tool=tool)
        u = usage(resp)
        total["input_tokens"] += u["input_tokens"] or 0
        total["output_tokens"] += u["output_tokens"] or 0

        try:
            out = tool_output(resp)
        except HTTPException:
            out = {}

        label = out.get("label")
        conf = out.get("confidence")
        valid = (
            label in req.labels
            and isinstance(conf, (int, float))
            and 0 <= conf <= 1
        )
        if valid:
            return {**out, "attempts": attempt + 1, "usage": total}

        # tell the model what went wrong and try again
        user = (
            f"{req.text}\n\n"
            f"(Your previous answer used the label {label!r}, which is not allowed. "
            f"Choose one of: {allowed}.)"
        )

    raise HTTPException(
        status_code=502,
        detail=f"Model returned an invalid label after {attempts} attempts: {label!r}",
    )