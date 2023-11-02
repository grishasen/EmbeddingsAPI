"""
API returning short embeddings for each sentence.

2023 Grigoriy Sen <grigoriy.sen@pega.com>

This file is part of the FastAPI Project.

This API accepts sentences as input and returns their corresponding embedding vectors.
Embedding vectors are numerical representations of text that capture semantic information,
enabling applications like sentiment analysis, text similarity, and more.
"""
import logging
import time
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


def get_logger(name, level=None) -> logging.Logger:
    logging.basicConfig(filename='embeddingsapi.log', filemode='a',
                        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    logger = logging.getLogger(name)
    handler = logging.handlers.RotatingFileHandler('embeddingsapi.log', maxBytes=1024 * 1024 * 10, backupCount=5)
    logger.addHandler(handler)
    if level is None:
        level = logging.DEBUG
    logger.setLevel(level)
    return logger


logger = get_logger(__name__)
model = SentenceTransformer('models/MiniLM-32dim-model')


class Sentences(BaseModel):
    sentences: List[str] = Field(examples=[['Tot 180,- retour op de Samsung Galaxy S23-serie',
                                            'Jusqu\'à 180,- remboursés sur la série Samsung Galaxy S23']])


class Embedding(BaseModel):
    sentence: str = Field(examples=['Tot 180,- retour op de Samsung Galaxy S23-serie',
                                    'Jusqu\'à 180,- remboursés sur la série Samsung Galaxy S23'])
    embedding: List[float]


class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name


app = FastAPI(
    title="API returning short embeddings for each sentence.",
    description="Build on Sentence Transformers and paraphrase-multilingual-MiniLM-L12-v2 model",
    version="0.1",
)


@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} exception."},
    )


@app.on_event("startup")
def startup_event():
    tic = time.perf_counter()
    toc = time.perf_counter()
    logger.info(f"Model loaded in {toc - tic:0.4f} seconds")


@app.get("/")
def read_root():
    return {"Name": "API returning short embeddings for each sentence."}


@app.post("/embeddings/")
async def embeddings(request: Sentences, response_model=list[Embedding]):
    """
        This is the root endpoint. When called will return embedding array for each sentence.

        Returns:
            array of dict: Each dict contains sentence and embedding.
    """
    logger.info(f"Start processing ")
    toc = time.perf_counter()
    embedz = model.encode(request.sentences, convert_to_tensor=True)
    tic = time.perf_counter()
    logger.info(f"Embeddings obtained in {tic - toc:0.4f} seconds")
    response = []
    for sentence, embedding in zip(request.sentences, embedz):
        emb = Embedding(sentence=sentence,
                        embedding=list(embedding))
        response.append(emb)
    logger.info(f"End processing")
    return response
