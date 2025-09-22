import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pydantic import BaseModel, Field
from fastapi import APIRouter,Depends
from pathlib import Path

router = APIRouter()

class KmIn(BaseModel):
    householdIncome: int
    leisurePurpose: int
    leisurePurpose2: int
    weekdayAvgLeisureTime: int
    weekendAvgLeisureTime: int
    restRecreationRate: int
    hobbyRate: int
    selfImprovementRate: int
    socialRelationshipRate: int
    leisureActivity1: int
    leisureActivity2: int
    leisureActivity3: int
    leisureActivity4: int
    leisureActivity5: int

# class KmOut(BaseModel):
#     animalName: str
#     animalType: str
#     description: str
#     animalDescription: str
#     clusterDescription: str
#     interesting: list[str]
#     # 디버그/확인용
#     cluster: int = Field(..., description="예측된 군집")
#     leisurePurpose_plus1: int = Field(..., description="leisurePurpose + 1 결과")
#     X_first_row: list[float] = Field(..., description="모델에 들어간 1행 벡터")

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

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "mz_processed.csv"   # ← 상대경로

train_df = pd.read_csv(DATA_FILE)

missing = [c for c in FEATURES if c not in train_df.columns]
if missing:
    raise ValueError(f"CSV에 다음 컬럼이 없습니다: {missing}")

X_train = train_df[FEATURES].astype(int)

k = 8
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),   # NaN 대비
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2, random_state=42)),
    ("kmeans", KMeans(n_clusters=k, n_init="auto", random_state=42)),
])
pipe.fit(X_train)

# @router.post("/kmeans", response_model = KmOut)
# def predict(body:KmIn):
#     x = np.array([[body.householdIncome, body.leisurePurpose, body.leisurePurpose2,
#                    body.weekdayAvgLeisureTime, body.weekendAvgLeisureTime, body.restRecreationRate,
#                    body.hobbyRate, body.selfImprovementRate, body.socialRelationshipRate,
#                    body.leisureActivity1, body.leisureActivity2, body.leisureActivity3,
#                    body.leisureActivity4, body.leisureActivity5]],
#                  dtype=int)
#     cluster = int(pipe.predict(x)[0])
#     return KmOut(vector=x.flatten().tolist(), cluster=cluster)

class TestOut(BaseModel):
    animalName: str
    animalType: str
    description: str
    animalDescription: str
    clusterDescription : str
    interesting : list[str]


def result_classify(cluster):
    if cluster == 0:
        resultvalue = {
            "animalName" : "당신의 동물은 재욱이입니다.",
            "animalType" : "수영구 붉은 수달",
            "description" : "당신은 빠르고 관찰력이 뛰어납니다.",
            "animalDescription" : "재욱이는 지능이 높은 동물입니다",
            "clusterDescription" : "당신의 유형은 다른 유형보다 스포츠/운동을 좋아합니다.",
            "interesting" : ["스피드", "붉은색", "BH하우스"]
        }
    elif cluster == 1:
        resultvalue = {
            "animalName": "당신의 동물은 상인이입니다.",
            "animalType": "범일동 붉은 매",
            "description": "당신은 파워가 뛰어납니다.",
            "animalDescription": "상인이는 파워가 높은 동물입니다",
            "clusterDescription": "당신의 유형은 다른 유형보다 스포츠/운동을 좋아합니다.",
            "interesting": ["파워", "붉은색", "오션브릿지"],
            # "test" : x[1]
        }
    elif cluster == 2:
        resultvalue = {
            "animalName": "당신의 동물은 지민이입니다.",
            "animalType": "아라관 푸른 수달",
            "description": "당신은 귀엽습니다.",
            "animalDescription": "지민이는 귀여운 동물입니다",
            "clusterDescription": "당신의 유형은 다른 유형보다 자기계발을 좋아합니다.",
            "interesting": ["cute", "푸른색", "아라관"]
        }
    elif cluster == 3:
        resultvalue = {
            "animalName": "당신의 동물은 예성이입니다.",
            "animalType": "영주동 흑곰",
            "description": "당신은 거대합니다.",
            "animalDescription": "예성이는 거대한 동물입니다",
            "clusterDescription": "당신의 유형은 다른 유형보다 사교/가족을 좋아합니다.",
            "interesting": ["덩치", "검은색", "코모도"]
        }
    elif cluster == 4:
        resultvalue = {
            "animalName": "당신의 동물은 민경이입니다.",
            "animalType": "동래역 1번 출구 지렁이",
            "description": "당신은 유연합니다.",
            "animalDescription": "민경이는 유연한 동물입니다",
            "clusterDescription": "당신의 유형은 다른 유형보다 여행/야외활동을 좋아합니다.",
            "interesting": ["몸치", "갈색", "동래어딘가"]
        }
    elif cluster == 5:
        resultvalue = {
            "animalName": "당신의 동물은 민이입니다.",
            "animalType": "청학동 주홍 매",
            "description": "당신은 수면의 질이 뛰어납니다.",
            "animalDescription": "민이는 빨리 자는 동물입니다",
            "clusterDescription": "당신의 유형은 다른 유형보다 일상/휴식을 좋아합니다.",
            "interesting": ["수면", "주홍색", "청학동"]
        }
    elif cluster == 6:
        resultvalue = {
            "animalName": "당신의 동물은 민서입니다.",
            "animalType": "대신동 노란 도마뱀",
            "description": "당신은 귀찮음이 뛰어납니다.",
            "animalDescription": "민서는 귀찮음 높은 동물입니다",
            "clusterDescription": "당신의 유형은 다른 유형보다 기타를 좋아합니다.",
            "interesting": ["귀찮다 이제", "노랑색", "대신동"]
        }
    else:
        resultvalue = {
            "animalName": "당신의 동물은 준용이입니다.",
            "animalType": "중리 회색 전봇대",
            "description": "당신은 길이가 뛰어납니다.",
            "animalDescription": "준영이는 키가 큰 동물입니다",
            "clusterDescription": "당신의 유형은 다른 유형보다 아 이제 귀찮다을 좋아합니다.",
            "interesting": ["이 사람 키가 크다", "회색", "중리"]
        }
    return resultvalue

@router.post("/v1/questions")
async def clusteranalyze(body: KmIn):
    x = np.array([[body.householdIncome, body.leisurePurpose, body.leisurePurpose2,
                   body.weekdayAvgLeisureTime, body.weekendAvgLeisureTime, body.restRecreationRate,
                   body.hobbyRate, body.selfImprovementRate, body.socialRelationshipRate,
                   body.leisureActivity1, body.leisureActivity2, body.leisureActivity3,
                   body.leisureActivity4, body.leisureActivity5]],
                 dtype=int) #설문조사 값 받아오기
    cluster = int(pipe.predict(x)[0]) #모델 실행 후 클러스터 값 저장
    analyze = result_classify(cluster)
    return analyze