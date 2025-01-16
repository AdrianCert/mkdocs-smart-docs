import logging
from pathlib import Path
from threading import Thread
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the provided classes
from .nlp import Model, QuestionAnsweringSupervisor

app = FastAPI()
logger = logging.getLogger("QA_Server")
logging.basicConfig(level=logging.INFO)


origins = [
    "http://localhost",  # Allow localhost
    "http://localhost:3000",  # If you're using React or another framework on port 3000, for example
    "*",  # Wildcard, allows all origins (use with caution in production)
]

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Set the allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


models: Dict[str, Model] = {}
document_storage: Dict[str, List[str]] = {}
supervisor = QuestionAnsweringSupervisor(logger=logger)


class DocumentRequest(BaseModel):
    model_id: str
    documents: List[str]


class TrainRequest(BaseModel):
    model_id: str
    epochs: int = 10
    batch_size: int = 4


class QuestionRequest(BaseModel):
    model_name: str
    question: str


def load_current_models():
    for model_path in Path("./.models").iterdir():
        model = Model.load(model_path)
        models[model_path.name] = model
        logger.info(f"Loaded model '{model_path.name}' from {model_path}")


@app.on_event("startup")
def startup():
    logger.info("Starting QA server...")
    load_current_models()


@app.post("/ingest")
def ingest_documents(request: DocumentRequest):
    try:
        if request.model_id not in document_storage:
            document_storage[request.model_id] = []
        document_storage[request.model_id].extend(request.documents)
        logger.info(
            f"Ingested {len(request.documents)} documents for model '{request.model_id}'."
        )
        return {
            "message": f"{len(request.documents)} documents ingested successfully for model '{request.model_id}'."
        }
    except Exception as e:
        logger.error(f"Error ingesting documents for model '{request.model_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {e}")


@app.post("/train")
def train_model(request: TrainRequest):
    if (
        request.model_id not in document_storage
        or not document_storage[request.model_id]
    ):
        raise HTTPException(
            status_code=400,
            detail=f"No documents available for training for model '{request.model_id}'. Please ingest documents first.",
        )

    model_name = request.model_id
    logger.info(f"Starting training for model: {model_name}")
    try:
        model = supervisor.train(
            *document_storage[request.model_id],
            epochs=request.epochs,
            batch_size=request.batch_size,
        )
        models[model_name] = model
        model_path = Path(f"./models/{model_name}")
        model.save(model_path)
        document_storage[request.model_id].clear()  # Clear documents after training
        logger.info(f"Model {model_name} saved at {model_path}")
        return {
            "message": f"Model '{model_name}' trained and saved.",
            "model_name": model_name,
        }
    except Exception as e:
        logger.error(f"Error training model '{model_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.post("/answer")
def answer_question(request: QuestionRequest):
    model_name = request.model_name
    question = request.question

    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    try:
        model = models[model_name]
        answer = model.answer(question)
        logger.info(f"Answered question with model '{model_name}': {answer}")
        return {"question": question, "answer": answer}
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}")


@app.post("/shutdown")
def shutdown_server():
    def stop_server():
        logger.info("Shutting down the server...")

    Thread(target=stop_server).start()
    return {"message": "Server is shutting down..."}


@app.on_event("shutdown")
def shutdown():
    logger.info("Server has been shut down.")
