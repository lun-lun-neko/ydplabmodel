from fastapi import FastAPI
from app.model import kmeansmodel, questionlist, submodel
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.include_router(kmeansmodel.router)
app.include_router(questionlist.router)
app.include_router(submodel.router)
app.mount("/static", StaticFiles(directory="static", html=False), name="static")

# @app.get("/")
# def read_root():
#     return {"message": ",,"}

@app.get("/")
def ping():
    return {"status": "ok"}

print("hello")