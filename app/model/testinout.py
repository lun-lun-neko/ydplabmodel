import numpy as np
from fastapi import APIRouter,Depends
from pydantic import BaseModel

router = APIRouter()

class irisin(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class irisout(BaseModel):
    vector:list[float]
    cluster: int


@router.post("/testinout")
def getdata(body: irisin):
    x = np.array([[body.sepal_length, body.sepal_width, body.petal_length, body.petal_width]], dtype=float)

    return {"vector": x.flatten().tolist()}