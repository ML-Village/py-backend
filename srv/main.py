import os
import sys
import logging
import subprocess
import traceback
from dotenv import load_dotenv, find_dotenv
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel


load_dotenv(find_dotenv())
# flags to only allow for only one proof
# the server cannot accomodate more than one proof
loaded_onnxmodel = None
loaded_inputdata = None
loaded_proofname = None
running = False

WEBHOOK_PORT = int(os.environ.get("PORT", 8888))  # 443, 80, 88 or 8443 (port need to be 'open')
WEBHOOK_LISTEN = '0.0.0.0' 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = FastAPI()


# allow cors
app.add_middleware(
    CORSMiddleware,
    #allow_origins=origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)


@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=WEBHOOK_LISTEN,
        port=WEBHOOK_PORT
    )