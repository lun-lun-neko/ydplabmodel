import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pydantic import BaseModel

from fastapi import APIRouter,Depends

router = APIRouter()

class irisin(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class irisout(BaseModel):
    vector:list[float]
    cluster: int

#data check
data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
df['class_name'] = df['target'].apply(lambda idx : data['target_names'][idx])
#print(df)

X, y = load_iris(return_X_y=True)

#기본적인 kmeans
X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=3, n_init="auto", random_state=42)
labels = kmeans.fit_predict(X_scaled)

#pipeline화
k = 3
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # NaN 대비
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=k, n_init="auto", random_state=42)),
])
pipe.fit(X)

@router.post("/predict", response_model=irisout)
def predict(body: irisin):
    x = np.array([[body.sepal_length, body.sepal_width, body.petal_length, body.petal_width]], dtype=float)
    cluster = int(pipe.predict(x)[0])
    return irisout(vector=x.flatten().tolist(), cluster=cluster)


