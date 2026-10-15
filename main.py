import os
import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
app = FastAPI(title="LLM playground")


