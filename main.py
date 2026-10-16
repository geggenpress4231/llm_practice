from fastapi import FastAPI
from routes import classify, extract, summarise

app = FastAPI(title="LLM playground")

app.include_router(classify.router)
app.include_router(extract.router)
app.include_router(summarise.router)