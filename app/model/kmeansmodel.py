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
            "animalName": "당신의 동물은 범고래입니다",
            "animalType": "위풍당당 범고래",
            "description": "당신은 스스로 삶을 개척하며 끊임없이 성장하는 사람입니다. 건강 관리와 취미, 자기계발로 자신을 단련하고 삶의 질을 높여갑니다.",
            "animalDescription": "범고래는 바다의 최상위 포식자답게 무리를 지어 역할을 분담하고, "
                                 "주변 지형과 조류를 영리하게 활용하며, "
                                 "심지어 인공적으로 파도를 일으켜 먹잇감을 사냥하는 등, "
                                 "매우 지능적인 전략을 구사합니다. "
                                 "이들은 최대 시속 56km로 바다를 가르고 하루에 100km 이상을 "
                                 "거침없이 유영할 수 있는 빠르고 강력한 신체 능력을 자랑합니다.",
            "typeDescriptoon": "당신은 푸른바다거북처럼 나만의 호흡으로 꾸준히 멀리 나아갑니다. "
                               "혼자 머무는 휴식에서 먼저 에너지를 채우고, 낮잠·카페 한 잔·온천/찜질 같은 일상적 재충전으로 자기만족의 소확행을 안정적으로 쌓아갑니다. "
                               "휴식은 도피가 아니라 다음 몰입을 위한 현명한 재충전, 과시는 내려두고 깊이와 지속성을 중시합니다.",
            "analSummary": [
                {
                    "subtitle": "💤 휴식 우선 전략",
                    "content": "‘마음의 안정·휴식을 우선시하는 유형입니다! "
                               "이 유형에서는 ‘레저시간 사용 목적 1,2순위’에서 마음의 안정·휴식이 각각 31.4%, 27.8%로 가장 높았습니다!"
                },
                {
                    "subtitle": "🌙 단독 몰입",
                    "content": "유형 중에서 ‘레저시간 중 대인관계·교제 사용 비율’이 평균적으로 가장 낮음과 동시에 "
                               "휴식/오락 비율'이 평균 83.6%로 8개의 유형 중 가장 높습니다. "
                               "또한 여가 목적 순위 1,2위 모두 ‘마음의 안정/휴식’에 치중되어 있어 혼자만의 시간을 즐기는 유형입니다!"
                }
            ],
            "metrics": [
                {
                    "subtitle": "여가시간 중 대인관계·교제 사용 비율",
                    "content": "전체 8개 유형 중에서 평균값(약8.3%) 8위 (전체 평균 : 약 23%)"
                },
                {
                    "subtitle": "여가시간 중 휴식·오락 사용 비율",
                    "content": "전체 유형 중에서 평균값(약 84%) 1위 (전체 평균 : 약 42%)"
                },
                {
                    "subtitle": "주요 활동",
                    "content": "'일상/휴식' 비율이 평균 대비 압도적으로 높으며, 대부분의 활동 카테고리에서 '참여하는 활동이 없다'는 응답 비율이 월등히 높게 나타납니다."
                }
            ],
            "culsterInfo": [
                {"cNumber": "0"},
                {"animal": "범고래"},
                {"type": "위퐁당당"}
            ],
            "interesting": ["건강이 최고", "자기관리 끝판왕", "긍정적사고", "성장마인드", "꾸준함"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/orca.svg"
        }
    elif cluster == 1:
        resultvalue = {
            "animalName": "당신의 동물은 왜가리입니다.",
            "animalType": "고요한 탐험가 왜가리",
            "description": "당신은 스트레스를 풀기 위해 여행과 편안한 만남을 즐기는 사람입니다. 익숙한 사람들과 함께하며 활력을 얻고, 직접적인 경험을 통해 삶을 채워갑니다.",
            "animalDescription": "왜가리는 다양한 환경에 모습을 드러내지만, "
                                 "물고기를 사냥하기에 최적의 장소에 다다르면 "
                                 "미동 없이 오랜 시간 사냥감에 집중하는 뛰어난 인내심을 보여줍니다. "
                                 "이들은 평소에는 독립적으로 생활하지만, "
                                 "번식기가 되면 수백 마리가 한데 모여 거대한 군집을 이룹니다. "
                                 "하지만 자신의 영역을 침범하는 동족에게는 강력한 공격성을 보입니다.",
            "typeDescriptoon": "당신은 푸른바다거북처럼 나만의 호흡으로 꾸준히 멀리 나아갑니다. "
                               "혼자 머무는 휴식에서 먼저 에너지를 채우고, 낮잠·카페 한 잔·온천/찜질 같은 일상적 재충전으로 자기만족의 소확행을 안정적으로 쌓아갑니다. "
                               "휴식은 도피가 아니라 다음 몰입을 위한 현명한 재충전, 과시는 내려두고 깊이와 지속성을 중시합니다.",
            "analSummary": [
                {
                    "subtitle": "💤 휴식 우선 전략",
                    "content": "‘마음의 안정·휴식을 우선시하는 유형입니다! "
                               "이 유형에서는 ‘레저시간 사용 목적 1,2순위’에서 마음의 안정·휴식이 각각 31.4%, 27.8%로 가장 높았습니다!"
                },
                {
                    "subtitle": "🌙 단독 몰입",
                    "content": "유형 중에서 ‘레저시간 중 대인관계·교제 사용 비율’이 평균적으로 가장 낮음과 동시에 "
                               "휴식/오락 비율'이 평균 83.6%로 8개의 유형 중 가장 높습니다. "
                               "또한 여가 목적 순위 1,2위 모두 ‘마음의 안정/휴식’에 치중되어 있어 혼자만의 시간을 즐기는 유형입니다!"
                }
            ],
            "metrics": [
                {
                    "subtitle": "여가시간 중 대인관계·교제 사용 비율",
                    "content": "전체 8개 유형 중에서 평균값(약8.3%) 8위 (전체 평균 : 약 23%)"
                },
                {
                    "subtitle": "여가시간 중 휴식·오락 사용 비율",
                    "content": "전체 유형 중에서 평균값(약 84%) 1위 (전체 평균 : 약 42%)"
                },
                {
                    "subtitle": "주요 활동",
                    "content": "'일상/휴식' 비율이 평균 대비 압도적으로 높으며, 대부분의 활동 카테고리에서 '참여하는 활동이 없다'는 응답 비율이 월등히 높게 나타납니다."
                }
            ],
            "culsterInfo": [
                {"cNumber": "1"},
                {"animal": "왜가리"},
                {"type": "고요한 탐험가"}
            ],
            "interesting": ["찐친파?", "안정감 추구", "리프레시"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/greyheron.svg"
        }
    elif cluster == 2:
        resultvalue = {
            "animalName": "당신의 동물은 매너티입니다.",
            "animalType": "여유만만 매너티",
            "description": "당신은 휴식과 여유를 사랑하는 사람입니다. 스스로를 돌보고 차분히 쉬는 시간을 즐기며, 가끔 여행으로 새로운 활력을 얻습니다.",
            "animalDescription": "매너티는 따뜻하고 얕은 수역의 풍부한 수초 군락지 근처에 서식하고, "
                                 "다 성장한 매너티는 거대한 몸집 덕분에 천적이 거의 없어 "
                                 "생존을 위한 스트레스 없이 평온한 삶을 살아갑니다. "
                                 "또한, 신진대사율이 매우 낮아 한 번 축적된 지방과 에너지로도 "
                                 "오랜 시간을 견딜 수 있기에 서두를 필요가 전혀 없으며, "
                                 "삶 자체가 곧 휴식이라고 할 수 있습니다.",
            "typeDescriptoon": "당신은 푸른바다거북처럼 나만의 호흡으로 꾸준히 멀리 나아갑니다. "
                               "혼자 머무는 휴식에서 먼저 에너지를 채우고, 낮잠·카페 한 잔·온천/찜질 같은 일상적 재충전으로 자기만족의 소확행을 안정적으로 쌓아갑니다. "
                               "휴식은 도피가 아니라 다음 몰입을 위한 현명한 재충전, 과시는 내려두고 깊이와 지속성을 중시합니다.",
            "analSummary": [
                {
                    "subtitle": "💤 휴식 우선 전략",
                    "content": "‘마음의 안정·휴식을 우선시하는 유형입니다! "
                               "이 유형에서는 ‘레저시간 사용 목적 1,2순위’에서 마음의 안정·휴식이 각각 31.4%, 27.8%로 가장 높았습니다!"
                },
                {
                    "subtitle": "🌙 단독 몰입",
                    "content": "유형 중에서 ‘레저시간 중 대인관계·교제 사용 비율’이 평균적으로 가장 낮음과 동시에 "
                               "휴식/오락 비율'이 평균 83.6%로 8개의 유형 중 가장 높습니다. "
                               "또한 여가 목적 순위 1,2위 모두 ‘마음의 안정/휴식’에 치중되어 있어 혼자만의 시간을 즐기는 유형입니다!"
                }
            ],
            "metrics": [
                {
                    "subtitle": "여가시간 중 대인관계·교제 사용 비율",
                    "content": "전체 8개 유형 중에서 평균값(약8.3%) 8위 (전체 평균 : 약 23%)"
                },
                {
                    "subtitle": "여가시간 중 휴식·오락 사용 비율",
                    "content": "전체 유형 중에서 평균값(약 84%) 1위 (전체 평균 : 약 42%)"
                },
                {
                    "subtitle": "주요 활동",
                    "content": "'일상/휴식' 비율이 평균 대비 압도적으로 높으며, 대부분의 활동 카테고리에서 '참여하는 활동이 없다'는 응답 비율이 월등히 높게 나타납니다."
                }
            ],
            "culsterInfo": [
                {"cNumber": "2"},
                {"animal": "매너티"},
                {"type": "여유만만"}
            ],
            "interesting": ["힐링이 필요해", "커피한잔의 여유", "자기돌봄", "느긋함"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/manatee.svg"
        }
    elif cluster == 3:
        resultvalue = {
            "animalName": "당신의 동물은 알바트로스입니다.",
            "animalType": "끈기의 비상가 알바트로스",
            "description": "당신은 짧은 여가시간에도 운동과 여행으로 활력을 찾는 사람입니다. 휴식보다는 몸을 움직이고 새로운 경험을 즐기며 에너지를 충전합니다.",
            "animalDescription": "알바트로스는 바람의 흐름을 완벽하게 파악하여 무작정 힘을 쏟기보다 효율적으로 비행하는 조류입니다. "
                                 "거의 날갯짓 없이 수천 킬로미터를 이동하며, 삶의 90% 이상을 육지에 착륙하지 않고 광활한 바다 위에서 생활합니다. "
                                 "특히, 한번 비행을 시작하면 먹고 자는 모든 활동을 하늘에서 해결하며 "
                                 "지구를 몇 바퀴나 돌 정도로 놀라운 비행력을 자랑합니다.",
            "typeDescriptoon": "당신은 푸른바다거북처럼 나만의 호흡으로 꾸준히 멀리 나아갑니다. "
                               "혼자 머무는 휴식에서 먼저 에너지를 채우고, 낮잠·카페 한 잔·온천/찜질 같은 일상적 재충전으로 자기만족의 소확행을 안정적으로 쌓아갑니다. "
                               "휴식은 도피가 아니라 다음 몰입을 위한 현명한 재충전, 과시는 내려두고 깊이와 지속성을 중시합니다.",
            "analSummary": [
                {
                    "subtitle": "💤 휴식 우선 전략",
                    "content": "‘마음의 안정·휴식을 우선시하는 유형입니다! "
                               "이 유형에서는 ‘레저시간 사용 목적 1,2순위’에서 마음의 안정·휴식이 각각 31.4%, 27.8%로 가장 높았습니다!"
                },
                {
                    "subtitle": "🌙 단독 몰입",
                    "content": "유형 중에서 ‘레저시간 중 대인관계·교제 사용 비율’이 평균적으로 가장 낮음과 동시에 "
                               "휴식/오락 비율'이 평균 83.6%로 8개의 유형 중 가장 높습니다. "
                               "또한 여가 목적 순위 1,2위 모두 ‘마음의 안정/휴식’에 치중되어 있어 혼자만의 시간을 즐기는 유형입니다!"
                }
            ],
            "metrics": [
                {
                    "subtitle": "여가시간 중 대인관계·교제 사용 비율",
                    "content": "전체 8개 유형 중에서 평균값(약8.3%) 8위 (전체 평균 : 약 23%)"
                },
                {
                    "subtitle": "여가시간 중 휴식·오락 사용 비율",
                    "content": "전체 유형 중에서 평균값(약 84%) 1위 (전체 평균 : 약 42%)"
                },
                {
                    "subtitle": "주요 활동",
                    "content": "'일상/휴식' 비율이 평균 대비 압도적으로 높으며, 대부분의 활동 카테고리에서 '참여하는 활동이 없다'는 응답 비율이 월등히 높게 나타납니다."
                }
            ],
            "culsterInfo": [
                {"cNumber": "3"},
                {"animal": "알바트로스"},
                {"type": "끈기의 비상가"}
            ],
            "interesting": ["도전DNA", "갓생", "에너자이저"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/albatross.svg"
        }
    elif cluster == 4:
        resultvalue = {
            "animalName": "당신의 동물은 푸른바다거북입니다.",
            "animalType": "슬로 라이퍼 푸른바다거북",
            "description": "당신은 차분하고 독립적으로 일상을 채우는 사람입니다. 자신만의 취향이 분명하며, 문화생활과 휴식에서 에너지 충전을 느낍니다.",
            "animalDescription": "푸른바다거북은 물 표면에 몸을 둥둥 띄워 햇볕을 쬐는 것을 즐기며, "
                                 "이때는 에너지를 거의 소모하지 않고 평화롭게 떠 있습니다. "
                                 "광활한 해초밭을 유유히 오가며 평화로운 식사를 만끽하기도 합니다. "
                                 "몸을 청결하게 관리하고 싶을 때는 물고기 떼를 찾아가 기생충이나 이물질을 제거하며 "
                                 "가려움이나 불편함 같은 스트레스를 해소하기도 합니다.",
            "typeDescriptoon": "당신은 푸른바다거북처럼 나만의 호흡으로 꾸준히 멀리 나아갑니다. "
                               "혼자 머무는 휴식에서 먼저 에너지를 채우고, 낮잠·카페 한 잔·온천/찜질 같은 일상적 재충전으로 자기만족의 소확행을 안정적으로 쌓아갑니다. "
                               "휴식은 도피가 아니라 다음 몰입을 위한 현명한 재충전, 과시는 내려두고 깊이와 지속성을 중시합니다.",
            # "typeDescription": {
            #     "title": "동물 유형 소개",
            #     "content": "당신은 푸른바다거북처럼 나만의 호흡으로 꾸준히 멀리 나아갑니다. "
            #                "혼자 머무는 휴식에서 먼저 에너지를 채우고, 낮잠·카페 한 잔·온천/찜질 같은 일상적 재충전으로 자기만족의 소확행을 안정적으로 쌓아갑니다. "
            #                "휴식은 도피가 아니라 다음 몰입을 위한 현명한 재충전, 과시는 내려두고 깊이와 지속성을 중시합니다."
            # }, # <- 프론트에서 title 컨트롤하지 않는 경우 title content 분리
            "analSummary": [  # {"title": "분석 요약"},  # <- 프론트에서 컨트롤 시 필요 없음.
                {
                    "subtitle": "💤 휴식 우선 전략",
                    "content": "‘마음의 안정·휴식을 우선시하는 유형입니다! "
                               "이 유형에서는 ‘레저시간 사용 목적 1,2순위’에서 마음의 안정·휴식이 각각 31.4%, 27.8%로 가장 높았습니다!"  #content도 결론과 근거로 분류는 가능할듯? 결론 : ~한 유형입니다! 근거 : ~가 높슾니다
                },
                {
                    "subtitle": "🌙 단독 몰입",
                    "content": "유형 중에서 ‘레저시간 중 대인관계·교제 사용 비율’이 평균적으로 가장 낮음과 동시에 "
                               "휴식/오락 비율'이 평균 83.6%로 8개의 유형 중 가장 높습니다. "
                               "또한 여가 목적 순위 1,2위 모두 ‘마음의 안정/휴식’에 치중되어 있어 혼자만의 시간을 즐기는 유형입니다!"
                }
            ],
            "metrics": [  # {"title": "주요 지표"},  # <- 프로늩에서 컨트롤 시 필요 없음.
                {
                    "subtitle": "여가시간 중 대인관계·교제 사용 비율",
                    "content": "전체 8개 유형 중에서 평균값(약8.3%) 8위 (전체 평균 : 약 23%)"
                },
                {
                    "subtitle": "여가시간 중 휴식·오락 사용 비율",
                    "content": "전체 유형 중에서 평균값(약 84%) 1위 (전체 평균 : 약 42%)"
                },
                {
                    "subtitle": "주요 활동",
                    "content": "'일상/휴식' 비율이 평균 대비 압도적으로 높으며, 대부분의 활동 카테고리에서 '참여하는 활동이 없다'는 응답 비율이 월등히 높게 나타납니다."
                }
            ],
            "culsterInfo": [
                {"cNumber": "4"},
                {"animal": "푸른바다거북"},
                {"type": "슬로라이퍼"}
            ],  # 재사용 대비 만든 custerInfo 근데 이 구조가 맞나?
            "interesting": ["조용한 몰입", "감성루틴", "휴식이 최고"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/greenseaturtle.svg"
        }
    elif cluster == 5:
        resultvalue = {
            "animalName": "당신의 동물은 돌고래입니다.",
            "animalType": "소신파 돌고래",
            "description": "당신은 에너지와 회복의 리듬을 아는 사람입니다. 야외활동을 우선시하되, 미디어·일상 휴식으로 컨디션을 영리하게 관리합니다.",
            "animalDescription": "돌고래는 거대한 무리를 이루어 복잡한 사회를 형성하며, "
                                 "무리별로 '유행가'를 공유할 만큼 뛰어난 지능과 문화적 유연성을 지녔습니다. "
                                 "이들은 배가 만드는 물살을 효율적으로 이용하여 최소한의 에너지로 "
                                 "먼 거리를 이동하는 지혜로움도 갖추고 있습니다. "
                                 "또한, 단체로 파도를 타며 서핑을 즐기거나 해파리로 공놀이를 하는 등 유희적인 행동을 자주 보여줍니다.",
            "typeDescriptoon": "당신은 푸른바다거북처럼 나만의 호흡으로 꾸준히 멀리 나아갑니다. "
                               "혼자 머무는 휴식에서 먼저 에너지를 채우고, 낮잠·카페 한 잔·온천/찜질 같은 일상적 재충전으로 자기만족의 소확행을 안정적으로 쌓아갑니다. "
                               "휴식은 도피가 아니라 다음 몰입을 위한 현명한 재충전, 과시는 내려두고 깊이와 지속성을 중시합니다.",
            "analSummary": [
                {
                    "subtitle": "💤 휴식 우선 전략",
                    "content": "‘마음의 안정·휴식을 우선시하는 유형입니다! "
                               "이 유형에서는 ‘레저시간 사용 목적 1,2순위’에서 마음의 안정·휴식이 각각 31.4%, 27.8%로 가장 높았습니다!"
                },
                {
                    "subtitle": "🌙 단독 몰입",
                    "content": "유형 중에서 ‘레저시간 중 대인관계·교제 사용 비율’이 평균적으로 가장 낮음과 동시에 "
                               "휴식/오락 비율'이 평균 83.6%로 8개의 유형 중 가장 높습니다. "
                               "또한 여가 목적 순위 1,2위 모두 ‘마음의 안정/휴식’에 치중되어 있어 혼자만의 시간을 즐기는 유형입니다!"
                }
            ],
            "metrics": [
                {
                    "subtitle": "여가시간 중 대인관계·교제 사용 비율",
                    "content": "전체 8개 유형 중에서 평균값(약8.3%) 8위 (전체 평균 : 약 23%)"
                },
                {
                    "subtitle": "여가시간 중 휴식·오락 사용 비율",
                    "content": "전체 유형 중에서 평균값(약 84%) 1위 (전체 평균 : 약 42%)"
                },
                {
                    "subtitle": "주요 활동",
                    "content": "'일상/휴식' 비율이 평균 대비 압도적으로 높으며, 대부분의 활동 카테고리에서 '참여하는 활동이 없다'는 응답 비율이 월등히 높게 나타납니다."
                }
            ],
            "culsterInfo": [
                {"cNumber": "5"},
                {"animal": "돌고래"},
                {"type": "소신파"}
            ],
            "interesting": ["페이스메이커", "운동과 회복", "액티브밸런스?"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/dolphin.svg"
        }
    elif cluster == 6:
        resultvalue = {
            "animalName": "당신의 동물은 펭귄입니다.",
            "animalType": "균형의 수호자 펭귄",
            "description": "당신은 여유롭고 주도적으로 삶을 설계하는 사람입니다. 휴식·자기계발·관계를 균형 있게 조율하며, 주말을 활용하여 콘텐츠와 여행으로 마음을 채우고 삶을 풍요롭게 가꿉니다.",
            "animalDescription": "펭귄은 영하 수십 도에 달하는 극한의 추위를 견뎌내기 위해 "
                                 "서로의 몸을 밀착시켜 열 손실을 최소화하는 '허들링' 전술을 사용합니다. "
                                 "이들은 천적의 위협이 도사리는 혹독한 환경 속에서도, "
                                 "가시 돋친 혀와 강력한 턱을 활용하여 먹이를 현명하게 사냥해냅니다. "
                                 "또한, 돌멩이로 보금자리를 짓고 그 위에 돌멩이를 쌓아 올리는 행위를 통해 서로에 대한 애정을 확인하기도 합니다.",
            "typeDescriptoon": "당신은 푸른바다거북처럼 나만의 호흡으로 꾸준히 멀리 나아갑니다. "
                               "혼자 머무는 휴식에서 먼저 에너지를 채우고, 낮잠·카페 한 잔·온천/찜질 같은 일상적 재충전으로 자기만족의 소확행을 안정적으로 쌓아갑니다. "
                               "휴식은 도피가 아니라 다음 몰입을 위한 현명한 재충전, 과시는 내려두고 깊이와 지속성을 중시합니다.",
            "analSummary": [
                {
                    "subtitle": "💤 휴식 우선 전략",
                    "content": "‘마음의 안정·휴식을 우선시하는 유형입니다! "
                               "이 유형에서는 ‘레저시간 사용 목적 1,2순위’에서 마음의 안정·휴식이 각각 31.4%, 27.8%로 가장 높았습니다!"
                },
                {
                    "subtitle": "🌙 단독 몰입",
                    "content": "유형 중에서 ‘레저시간 중 대인관계·교제 사용 비율’이 평균적으로 가장 낮음과 동시에 "
                               "휴식/오락 비율'이 평균 83.6%로 8개의 유형 중 가장 높습니다. "
                               "또한 여가 목적 순위 1,2위 모두 ‘마음의 안정/휴식’에 치중되어 있어 혼자만의 시간을 즐기는 유형입니다!"
                }
            ],
            "metrics": [
                {
                    "subtitle": "여가시간 중 대인관계·교제 사용 비율",
                    "content": "전체 8개 유형 중에서 평균값(약8.3%) 8위 (전체 평균 : 약 23%)"
                },
                {
                    "subtitle": "여가시간 중 휴식·오락 사용 비율",
                    "content": "전체 유형 중에서 평균값(약 84%) 1위 (전체 평균 : 약 42%)"
                },
                {
                    "subtitle": "주요 활동",
                    "content": "'일상/휴식' 비율이 평균 대비 압도적으로 높으며, 대부분의 활동 카테고리에서 '참여하는 활동이 없다'는 응답 비율이 월등히 높게 나타납니다."
                }
            ],
            "culsterInfo": [
                {"cNumber": "6"},
                {"animal": "펭귄"},
                {"type": "균형의 수호자"}
            ],
            "interesting": ["균형의 미학", "삶의 조각가", "주말이 좋아"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/penguin.svg"
        }
    else:
        resultvalue = {
            "animalName": "당신의 동물은 해파리입니다.",
            "animalType": "유유자적 해파리",
            "description": "당신은 자신만의 평화를 지키며 여유롭게 삶을 보내는 사람입니다. 일상 속에서도 마음의 안정을 우선시하고, 스스로의 방식으로 스트레스를 풀어갑니다.",
            "animalDescription": "해파리는 플랑크톤의 일종으로, 스스로 헤엄칠 수 없어 바다의 흐름에 몸을 맡긴 채 이동합니다. "
                                 "뇌 없이 단순한 신경망만으로 세상을 감지하는 해파리는 주변 환경을 온몸으로 받아들이며 살아갑니다. "
                                 "또한, 신체 일부가 손상되어도 빠른 속도로 재생해내는 탁월한 회복력을 가지고 있습니다.",
            "typeDescriptoon": "당신은 푸른바다거북처럼 나만의 호흡으로 꾸준히 멀리 나아갑니다. "
                               "혼자 머무는 휴식에서 먼저 에너지를 채우고, 낮잠·카페 한 잔·온천/찜질 같은 일상적 재충전으로 자기만족의 소확행을 안정적으로 쌓아갑니다. "
                               "휴식은 도피가 아니라 다음 몰입을 위한 현명한 재충전, 과시는 내려두고 깊이와 지속성을 중시합니다.",
            "analSummary": [
                {
                    "subtitle": "💤 휴식 우선 전략",
                    "content": "‘마음의 안정·휴식을 우선시하는 유형입니다! "
                               "이 유형에서는 ‘레저시간 사용 목적 1,2순위’에서 마음의 안정·휴식이 각각 31.4%, 27.8%로 가장 높았습니다!"
                },
                {
                    "subtitle": "🌙 단독 몰입",
                    "content": "유형 중에서 ‘레저시간 중 대인관계·교제 사용 비율’이 평균적으로 가장 낮음과 동시에 "
                               "휴식/오락 비율'이 평균 83.6%로 8개의 유형 중 가장 높습니다. "
                               "또한 여가 목적 순위 1,2위 모두 ‘마음의 안정/휴식’에 치중되어 있어 혼자만의 시간을 즐기는 유형입니다!"
                }
            ],
            "metrics": [
                {
                    "subtitle": "여가시간 중 대인관계·교제 사용 비율",
                    "content": "전체 8개 유형 중에서 평균값(약8.3%) 8위 (전체 평균 : 약 23%)"
                },
                {
                    "subtitle": "여가시간 중 휴식·오락 사용 비율",
                    "content": "전체 유형 중에서 평균값(약 84%) 1위 (전체 평균 : 약 42%)"
                },
                {
                    "subtitle": "주요 활동",
                    "content": "'일상/휴식' 비율이 평균 대비 압도적으로 높으며, 대부분의 활동 카테고리에서 '참여하는 활동이 없다'는 응답 비율이 월등히 높게 나타납니다."
                }
            ],
            "culsterInfo": [
                {"cNumber": "7"},
                {"animal": "해파리"},
                {"type": "유유자적"}
            ],
            "interesting": ["슬기로운 집콕생활", "소확행", "내 방이 최고", "시간 만수르"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/jellyfish.svg"
        }
    return resultvalue

@router.post("/questions")
async def clusteranalyze(body: KmIn):
    lp1 = body.leisurePurpose+1
    x = np.array([[body.householdIncome, lp1, body.leisurePurpose2,
                   body.weekdayAvgLeisureTime, body.weekendAvgLeisureTime, body.restRecreationRate,
                   body.hobbyRate, body.selfImprovementRate, body.socialRelationshipRate,
                   body.leisureActivity1, body.leisureActivity2, body.leisureActivity3,
                   body.leisureActivity4, body.leisureActivity5]],
                 dtype=int) #설문조사 값 받아오기
    cluster = int(pipe.predict(x)[0]) #모델 실행 후 클러스터 값 저장
    analyze = result_classify(cluster)
    return analyze

