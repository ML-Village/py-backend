FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /srv
COPY . ./

# Install production dependencies.
RUN pip install tensorflow onnx onnxruntime numpy
RUN pip install -r requirements.txt

CMD python main.py
#CMD [ "bash" ]