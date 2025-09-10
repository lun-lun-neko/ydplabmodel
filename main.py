from fastapi import FastAPI
from app.model import testinout, testmodel, kmeansmodel

app = FastAPI()

app.include_router(testinout.router)
app.include_router(testmodel.router)
app.include_router(kmeansmodel.router)

@app.get("/")
def read_root():
    return {"message": ",,"}

print("hello")