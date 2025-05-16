from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from agent import process_query

app = FastAPI(title="LLM-Powered GraphQL Agent")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Query(BaseModel):
    q: str

class Response(BaseModel):
    answer: str

@app.post("/query", response_model=Response)
async def query_endpoint(query: Query):
    try:
        logger.info(f"Received query: {query.q}")
        answer = await process_query(query.q)
        return Response(answer=answer)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000) 