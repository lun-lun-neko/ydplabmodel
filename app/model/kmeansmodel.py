import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pydantic import BaseModel
from fastapi import APIRouter,Depends

router = APIRouter()

class KmIn(BaseModel):
    household_income: int
    leisure_purpose: int
    leisure_purpose_2: int
    weekday_avg_leisure_time: int
    weekend_avg_leisure_time: int
    rest_recreation_rate: int
    hobby_rate: int
    self_improvement_rate: int
    social_relationship_rate: int
    leisure_activity_1: int
    leisure_activity_2: int
    leisure_activity_3: int
    leisure_activity_4: int
    leisure_activity_5: int

class KmOut(BaseModel):
    vector: list[float]
    cluster: int

FEATURES = [
    "household_income",
    "leisure_purpose",
    "leisure_purpose_2",
    "weekday_avg_leisure_time",
    "weekend_avg_leisure_time",
    "rest_recreation_rate",
    "hobby_rate",
    "self_improvement_rate",
    "social_relationship_rate",
    "leisure_activity_1",
    "leisure_activity_2",
    "leisure_activity_3",
    "leisure_activity_4",
    "leisure_activity_5",
]

train_df = pd.read_csv("C:\ydplabfast\mz_processed.csv")

missing = [c for c in FEATURES if c not in train_df.columns]
if missing:
    raise ValueError(f"CSV에 다음 컬럼이 없습니다: {missing}")

X_train = train_df[FEATURES].astype(int)

k = 5
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # NaN 대비
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2, random_state=42)),
    ("kmeans", KMeans(n_clusters=k, n_init="auto", random_state=42)),
])
pipe.fit(X_train)

@router.post("/kmeans", response_model = KmOut)
def predict(body:KmIn):
    x = np.array([[body.household_income, body.leisure_purpose, body.leisure_purpose_2,
                   body.weekday_avg_leisure_time, body.weekend_avg_leisure_time, body.rest_recreation_rate,
                   body.hobby_rate, body.self_improvement_rate, body.social_relationship_rate,
                   body.leisure_activity_1, body.leisure_activity_2, body.leisure_activity_3,
                   body.leisure_activity_4, body.leisure_activity_5]],
                 dtype=int)
    cluster = int(pipe.predict(x)[0])
    return KmOut(vector=x.flatten().tolist(), cluster=cluster)


