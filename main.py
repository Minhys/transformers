from fastapi import FastAPI
from pydantic import BaseModel
from celery.result import AsyncResult
from celery_worker import create_music_task

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str

class TaskResponse(BaseModel):
    task_id: str

class StatusResponse(BaseModel):
    task_id: str
    status: str
    result: str | None = None


@app.post("/generate", response_model=TaskResponse)
async def generate_music(request: GenerationRequest):
    """
    Accepts a music generation prompt, launches a background task,
    and returns the task ID.
    """
    task = create_music_task.delay(request.prompt)
    return {"task_id": task.id}


@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """
    Checks the status of a generation task.
    """
    task_result = AsyncResult(task_id)

    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }

    if task_result.successful():
        response["result"] = task_result.result

    return response


@app.get("/")
async def root():
    return {"message": "Music Generation API with Celery is running."}
