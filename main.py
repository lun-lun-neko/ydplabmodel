from fastapi import FastAPI
from app.model import kmeansmodel, questionlist

app = FastAPI()

app.include_router(kmeansmodel.router)
app.include_router(questionlist.router)

# @app.get("/")
# def read_root():
#     return {"message": ",,"}

@app.get("/")
def ping():
    return {"status": "ok"}

print("hello")