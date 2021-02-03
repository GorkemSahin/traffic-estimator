from fastapi import FastAPI
import lstm
import sarima
import mlp
from pydantic import BaseModel
from typing import Dict, List
import json
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DateModel(BaseModel):
    year: int
    month: int
    day: int

class ResponseModel(BaseModel):
    date: int
    value: float

class LSTM(BaseModel):
    file: str
    num_epochs: int
    learning_rate: float
    num_layers: int
    neuron_count: int
    seq_length: int
    start: DateModel
    test_start: DateModel
    end: DateModel

class MLP(BaseModel):
    file: str
    num_epochs: int
    neuron_count: int
    learning_rate: float
    dropout_rate: float
    start: DateModel
    test_start: DateModel
    end: DateModel

class SARIMA(BaseModel):
    file: str
    p: int
    d: int
    q: int
    P: int
    D: int
    Q: int
    s: int
    start: DateModel
    test_start: DateModel
    end: DateModel

class Response(BaseModel):
    prediction: List[ResponseModel]
    data: List[ResponseModel]
    mape: float

@app.get("/")
async def root():
    #logic.test()
    return {"message": "Ahmet necdet sezer"}

@app.post("/lstm")
async def parse(body: LSTM):
    msg = lstm.main(
        file = body.file,
        num_epochs = body.num_epochs,
        learning_rate = body.learning_rate,
        num_layers = body.num_layers,
        hidden_size = body.neuron_count,
        seq_length = body.seq_length,
        start= body.start,
        test_start= body.test_start,
        end= body.end)

    prediction = json.loads(msg[0])
    data = json.loads(msg[1])
    print(prediction)
    print(data)
    mape = float(msg[2])
    print(mape)
    prediction = [ResponseModel(date=k, value=v) for k, v in prediction.items()]
    data = [ResponseModel(date=k, value=v) for k, v in data.items()]
    resp = Response(prediction=prediction, data=data, mape=mape)
    #print(resp)
    return resp

@app.post("/sarima")
async def parse(body: SARIMA):
    print(body)
    msg = sarima.main(
        file = body.file,
        p = body.p,
        d = body.d,
        q = body.q,
        P = body.P,
        D = body.D,
        Q = body.Q,
        s = body.s,
        start= body.start,
        test_start= body.test_start,
        end= body.end)


    prediction = json.loads(msg[0])
    data = json.loads(msg[1])
    print(prediction)
    print(data)
    mape = float(msg[2])
    prediction = [ResponseModel(date=k, value=v) for k, v in prediction.items()]
    data = [ResponseModel(date=k, value=v) for k, v in data.items()]
    resp = Response(prediction=prediction, data=data, mape=mape)
    #print(resp)
    return resp

@app.post("/mlp")
async def parse(body: MLP):
    print(body)
    msg = mlp.main(
        file = body.file,
        epoch = body.num_epochs,
        neuron = body.neuron_count,
        learning = body.learning_rate,
        dropout = body.dropout_rate,
        start= body.start,
        test_start= body.test_start,
        end= body.end)

    prediction = json.loads(msg[0])
    data = json.loads(msg[1])
    print("-------------------------")
    print("-------------------------")
    print("-------------------------")
    print("-------------------------")
    print(prediction)
    print("-------------------------")
    print("-------------------------")
    print("-------------------------")
    print("-------------------------")
    print(data)
    print("-------------------------")
    print("-------------------------")
    print("-------------------------")
    print("-------------------------")
    mape = float(msg[2])
    prediction = [ResponseModel(date=k, value=v) for k, v in prediction.items()]
    data = [ResponseModel(date=k, value=v) for k, v in data.items()]
    resp = Response(prediction=prediction, data=data, mape=mape)
    print(resp)
    return resp


@app.post("/test")
async def parse(body: LSTM):
    #parse_data.test(body.file)
    return body