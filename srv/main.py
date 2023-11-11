import os
import sys
import logging
import subprocess
import traceback
from dotenv import load_dotenv, find_dotenv
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import onnx
import onnxruntime as ort
import numpy as np

from fastapi import Query
from pydantic.types import Json
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

load_dotenv(find_dotenv())
# flags to only allow for only one proof
# the server cannot accomodate more than one proof
loaded_onnxmodel = None
loaded_inputdata = None
loaded_proofname = None
running = False

WEBHOOK_PORT = int(os.environ.get("PORT", 8888))  # 443, 80, 88 or 8443 (port need to be 'open')
WEBHOOK_LISTEN = '0.0.0.0' 

onnx_model = onnx.load("./models/ttt.onnx")

#print(onnx_model)
#print(onnx.checker.check_model(onnx_model))


sess = ort.InferenceSession("./models/ttt.onnx")
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)
input_type = sess.get_inputs()[0].type
print("Input type  :", input_type)

output_name = sess.get_outputs()[0].name
print("Output name  :", output_name)  
output_shape = sess.get_outputs()[0].shape
print("Output shape :", output_shape)
output_type = sess.get_outputs()[0].type
print("Output type  :", output_type)

# x = np.array([[2,2,2,1,2,2,2,2,2]]).astype("float32")
# # #x = np.random.rand(1,9)
# # x = x.astype(np.float32)
# # print(x)

# onnx_pred = sess.run([output_name], {input_name: x})
# # print("score:")
# print(onnx_pred[0][0][0])

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

@app.post("/sampleinfer")
async def sampleinfer():
    x = np.array([[2,2,2,1,2,2,2,2,2]]).astype("float32")
    onnx_pred = sess.run([output_name], {input_name: x})
    score = onnx_pred[0][0][0]
    return {"score": str(score)}


@app.post("/infer")
async def infer(request: Request):

    jsonpayload = await request.json()

    results_dict = {}
    for i, b in enumerate(jsonpayload):
        x = np.array([b]).astype("float32")
        onnx_pred = sess.run([output_name], {input_name: x})
        score = onnx_pred[0][0][0]
        results_dict[i] = score
    print(results_dict)
    sorted_results = sorted(results_dict.items(), key=lambda x:x[1], reverse=True)
    print(sorted_results)
    bestconfigkey = sorted_results[0][0]
    print(jsonpayload[bestconfigkey])

    return JSONResponse(content=jsonable_encoder(jsonpayload[bestconfigkey]))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=WEBHOOK_LISTEN,
        port=WEBHOOK_PORT
    )