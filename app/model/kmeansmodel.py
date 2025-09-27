import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends
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
DATA_FILE = BASE_DIR / "data" / "mz_processed.csv"  # ← 상대경로

train_df = pd.read_csv(DATA_FILE)

missing = [c for c in FEATURES if c not in train_df.columns]
if missing:
    raise ValueError(f"CSV에 다음 컬럼이 없습니다: {missing}")

X_train = train_df[FEATURES].astype(int)

k = 8
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # NaN 대비
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
    clusterDescription: str
    interesting: list[str]


def result_classify(cluster):
    if cluster == 0:
        resultvalue = {
            "animal": "범고래",
            "animalName": "당신의 동물은 범고래입니다",
            "animalType": "위풍당당 범고래",
            "description": "당신은 스스로 삶을 개척하며 끊임없이 성장하는 사람입니다. 건강 관리와 취미, 자기계발로 자신을 단련하고 삶의 질을 높여갑니다.",
            "animalDescription": "범고래는 바다의 최상위 포식자답게 무리를 지어 역할을 분담하고, 주변 지형과 조류를 영리하게 활용하며, 심지어 인공적으로 파도를 일으켜 먹잇감을 사냥하는 등, 매우 지능적인 전략을 구사합니다. 이들은 최대 시속 56km로 바다를 가르고 하루에 100km 이상을 거침없이 유영할 수 있는 빠르고 강력한 신체 능력을 자랑합니다.",
            "typeDescriptoon": "당신은 무리를 이끄는 범고래처럼 현재의 성공에 안주하지 않고, 끝없이 새로운 영역을 탐험하며 삶을 개척하는 리더입니다. 여가 시간을 단순한 휴식이 아닌 건강 관리와 성장을 위한 투자로 여기며, 평균보다 적은 시간임에도 불구하고 스포츠와 자기계발 같은 생산적인 활동으로 알차게 채웁니다. 이러한 삶의 태도는 당신이 삶의 주도권을 쥐고, 자신을 단련하며 균형 잡힌 최상의 삶을 만들고자 하는 강한 의지를 명확히 보여줍니다.",
            "analSummary": [
                {
                    "subtitle": "💪 운동 중심의 에너지 충전",
                    "content": "이 유형은 운동을 통해 에너지를 충전하는 타입입니다. '관심 여가활동 1~5순위'에서는 [스포츠/운동] 비율이 가장 높았으며, '여가시간 사용 목적 1,2순위'에서도 [건강 관리]가 압도적으로 높았습니다.",
                    "metrics": [
                        {
                            "subtitle": "관심 여가 활동",  # 💪 운동 중심의 에너지 충전
                            "content": "이 유형은 '관심 여가 활동 1,2 순위'에서 [스포츠/운동]을 선택한 비율이 32.7%로 전체 유형 평균인 16.5% 대비 2배 수준입니다. 해당 지표는 모든 유형 중 1위에 해당합니다."
                        },
                        {
                            "subtitle": "여가시간 사용 목적",  # 💪 운동 중심의 에너지 충전
                            "content": "이 유형은 '여가시간 사용 목적 1순위'에서 [건강 관리]를 선택한 비율이 27.3%로, 전체 유형 평균인 12.7% 대비 두 배 이상 높습니다. 해당 지표는 모든 유형 중 1위에 해당합니다. 또한, '사용 목적 2순위'에서 [건강 관리]를 선택한 비율이 19.3%로, 전체 유형 평균인 11.5% 대비 1.7배 수준입니다."
                        }
                    ]
                },
                {
                    "subtitle": "🙅‍♂️ 단순 휴식은 놉",
                    "content": "'여가 시간 중 휴식·오락 비율'은 전체 유형 평균보다 훨씬 낮았습니다. 대신 '취미, 본인계발, 대인관계·교제' 비율이 모두 전체 유형 평균보다 높아 적극적이고 성장 지향적인 여가 활용을 보여줍니다.",
                    "metrics": [
                        {
                            "subtitle": "여가시간 중 휴식·오락 사용 비율",  # 🙅‍♂️ 단순 휴식은 놉!
                            "content": "이 유형은 '여가 시간 중 휴식·오락 활용 비율'의 평균이 16.8%로, 전체 유형 평균인 42.2% 대비 0.4배 수준입니다. 순위는 전체 유형 중 7위로, 휴식에 가장 적은 시간을 할애하는 편입니다."
                        },
                        {
                            "subtitle": "여가시간 중 취미, 자기계발, 대인관계·교제 사용 비율",  # 🙅‍♂️ 단순 휴식은 놉!
                            "content": "이 유형은 '여가시간 중 사용 비율'에서 취미(33.4%), 본인계발(21.4%), 대인관계·교제(27%)로 세가지 비율이 전체 유형 대비 높아, 적극적이고 성장 지향적인 여가 활용 특성을 보입니다."
                        }
                    ]
                }
            ],
            "culsterInfo": [
                {"cNumber": "0"},
                {"type": "위퐁당당"}
            ],
            "interesting": ["건강지키미", "자기관리끝판왕", "긍정적사고", "성장마인드", "꾸준함"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/orca.png"
        }
    elif cluster == 1:
        resultvalue = {
            "animal": "왜가리",
            "animalName": "당신의 동물은 왜가리입니다.",
            "animalType": "고요한 탐험가 왜가리",
            "description": "당신은 스트레스를 풀기 위해 여행과 편안한 만남을 즐기는 사람입니다. 익숙한 사람들과 함께하며 활력을 얻고, 직접적인 경험을 통해 삶을 채워갑니다.",
            "animalDescription": "왜가리는 다양한 환경에 모습을 드러내지만, "
                                 "물고기를 사냥하기에 최적의 장소에 다다르면 "
                                 "미동 없이 오랜 시간 사냥감에 집중하는 뛰어난 인내심을 보여줍니다. "
                                 "이들은 평소에는 독립적으로 생활하지만, "
                                 "번식기가 되면 수백 마리가 한데 모여 거대한 군집을 이룹니다. "
                                 "하지만 자신의 영역을 침범하는 동족에게는 강력한 공격성을 보입니다.",
            "typeDescriptoon": "당신은 왜가리처럼 조용한 물가에서 자신만의 리듬으로 하루를 살아가는 사람입니다. 혼자만의 시간을 보내기보다는 편안하고 익숙한 사람들과 함께 스트레스를 해소하는 것을 중요하게 생각합니다. 몸과 마음의 스트레스를 푸는 것을 가장 중요하게 생각하며, 함께하는 즐거움을 통해 에너지를 재충전합니다.",
            "analSummary": [
                {
                    "subtitle": "🧑‍🤝‍🧑 교제·야외활동 선호",
                    "content": "'여가 시간 중 대인관계·교제 사용 비율'이 모든 유형 중 세 번째로 높습니다. 또한 관심 여가 활동 1순위에서 [여행/야외활동] 비율이 가장 높고 다른 유형들과 비교했을 때도 가장 높기 때문에, 주변의 친한 친구들과 야외활동 하는 것을 즐기는 유형입니다!",
                    "metrics": [
                        {
                            "subtitle": "여가시간 중 대인관계·교제 사용 비율",  # 🧑‍🤝‍🧑교제·야외활동 선호
                            "content": "여가시간 중 대인관계·교제 사용 비율은 27.1%로 모든 유형 중에서 3위었습니다. 또한 전체 유형의 평균 비율인 23.2%에 비해 1.2배 수준입니다."
                        },
                        {
                            "subtitle": "관심 여가활동 1순위",  # 🧑‍🤝‍🧑교제·야외활동 선호
                            "content": "'관심 여가 활동 1순위'가 [여행/야외활동]으로 나타났으며 비율은 24.6%었습니다. 이는 모든 유형 중에서도 1위입니다!"
                        }
                    ]
                },
                {
                    "subtitle": "🌈 여가 목적의 다양성",
                    "content": "이 유형은 여가시간 사용 목적 1순위로 [스트레스 해소], [가족·지인과의 시간], [자기만족·즐거움], [마음의 안정·휴식]의 항목을 다양하게 선택하였습니다. 그러므로 목적의 1순위는 개인별로 상이하여 특정 항목에 집중되기보단 다양한 양상이 보여집니다!",
                    "metrics": [
                        {
                            "subtitle": "여가시간 사용 목적 1순위",  # 🌈 여가 목적의 다양성
                            "content": "'여가시간 사용 목적 1순위'에서 스트레스 해소(20.7%), 가족·지인 등과 시간(20.4%), 자기만족(17.2%), 마음의 안정·휴식(16.3%) 등으로 비율이 크게 차이가 나지 않아 이 유형에서는 여러 목적들을 고려해볼 수 있습니다."
                        }
                    ]
                },
                {
                    "subtitle": "🌿 휴식과 관계 중시",
                    "content": "이 유형은 '여가시간 사용 목적 2순위'로 [마음의 안정·휴식]이 1위, [가족·지인과의 시간]이 2위입니다. 이때, [마음의 안정·휴식], [가족·지인과의 시간]은 1순위와 2순위 모두 비율이 높은 편에 속하므로 다른 유형에 비해 마음의 안정·휴식 및 가족·지인과의 시간을 가장 중요시하는 유형인 것으로 보입니다!",
                    "metrics": [
                        {
                            "subtitle": "여가시간 사용 목적 2순위",  # 🌿 휴식과 관계 중시
                            "content": "'여가시간 사용 목적 2순위'에서 [마음의 안정·휴식]이 27.5%, [가족·지인과의 시간]이 19.2%로 집중되어있으며, 각각의 전체 유형 평균인 17.9%와 14.8%에 비해 1.5배, 1.3배 수준입니다. "
                        }
                    ]
                }
            ],
            "culsterInfo": [
                {"cNumber": "1"},
                {"type": "고요한 탐험가"}
            ],
            "interesting": ["힐링중시", "안정적교류", "야외활동선호", "평온추구", "에너지재충전"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/greyheron.png"
        }
    elif cluster == 2:
        resultvalue = {
            "animal": "매너티",
            "animalName": "당신의 동물은 매너티입니다.",
            "animalType": "여유만만 매너티",
            "description": "당신은 휴식과 여유를 사랑하는 사람입니다. 스스로를 돌보고 차분히 쉬는 시간을 즐기며, 가끔 여행으로 새로운 활력을 얻습니다.",
            "animalDescription": "매너티는 따뜻하고 얕은 수역의 풍부한 수초 군락지 근처에 서식하고, "
                                 "다 성장한 매너티는 거대한 몸집 덕분에 천적이 거의 없어 "
                                 "생존을 위한 스트레스 없이 평온한 삶을 살아갑니다. "
                                 "또한, 신진대사율이 매우 낮아 한 번 축적된 지방과 에너지로도 "
                                 "오랜 시간을 견딜 수 있기에 서두를 필요가 전혀 없으며, "
                                 "삶 자체가 곧 휴식이라고 할 수 있습니다.",
            "typeDescriptoon": "당신은 매너티처럼 풍요로운 환경 속의 여유만만 라이프를 추구합니다. 주변의 자원을 지혜롭게 활용하며, 거창한 모험이나 빠른 성취에 연연하지 않고, 커피 한 잔, 좋아하는 책, 혹은 햇살이 잘 드는 공원 벤치 등 소소하지만 확실한 행복을 통해 활력을 얻습니다. 주변의 위협이나 외부의 기대에 흔들리지 않고, 느긋한 리듬 속에서 자신의 삶을 여유롭게 유영하는 유형입니다.",
            "analSummary": [
                {
                    "subtitle": "🛌 휴식이 최고",
                    "content": "'여가 시간 중 휴식·오락 사용 비율'이 상당히 높군요. 또한 '관심 여가 활동 1순위'도 [일상/휴식] 선택 비율이 가장 높습니다. 당신은 편안한 휴식과 일상적 즐거움을 선호하는 유형이네요!",
                    "metrics": [
                        {
                            "subtitle": "여가시간 중 휴식·오락 사용 비율",  # 🛌 휴식이 최고
                            "content": "이 유형은 '여가시간 중 휴식·오락 사용 비율'의 평균값이 57.2%로 전체 유형 중 세 번째로 높습니다. 이는 전체 유형의 평균 비율인 42.2%의 1.4배 수준입니다."
                        },
                        {
                            "subtitle": "관심 여가활동",  # 🛌 휴식이 최고
                            "content": "이 유형의 '관심 여가활동 1순위' 선택 비율은 [일상/휴식]이 30.4%로 가장 높습니다. 이는 모든 유형 중 세 번째로 높은 비율을 나타냅니다. 다음으로는 [미디어/콘텐츠]가 24.5%로 두 번째로 높습니다."
                        }
                    ]
                },
                {
                    "subtitle": "🧘마음의 안정·휴식 우선",
                    "content": "이 유형은 '여가시간 사용 목적 1순위'로 [마음의 안정·휴식]이 가장 높습니다. 여가의 목적이 휴식인 만큼 자기 계발보다는 마음의 안정과 휴식이 우선인 사람이군요!",
                    "metrics": [
                        {
                            "subtitle": "여가시간 사용 목적",  # 🧘 마음의 안정·휴식 우선
                            "content": "이 유형의 '여가시간 사용 목적 1순위'에서 [마음의 안정·휴식]선택 비율이 34.4%로 가장 높습니다. 이는 전체 유형 중 두 번째로 높은 비율을 나타냅니다."
                        }
                    ]
                },
            ],
            "culsterInfo": [
                {"cNumber": "2"},
                {"type": "여유만만"}
            ],
            "interesting": ["안정우선", "휴식집중형", "느긋함", "회복지향", "소확행"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/manatee.png"
        }
    elif cluster == 3:
        resultvalue = {
            "animal": "알바트로스",
            "animalName": "당신의 동물은 알바트로스입니다.",
            "animalType": "끈기의 비상가 알바트로스",
            "description": "당신은 짧은 여가시간에도 운동과 여행으로 활력을 찾는 사람입니다. 휴식보다는 몸을 움직이고 새로운 경험을 즐기며 에너지를 충전합니다.",
            "animalDescription": "알바트로스는 바람의 흐름을 완벽하게 파악하여 무작정 힘을 쏟기보다 효율적으로 비행하는 조류입니다. "
                                 "거의 날갯짓 없이 수천 킬로미터를 이동하며, 삶의 90% 이상을 육지에 착륙하지 않고 광활한 바다 위에서 생활합니다. "
                                 "특히, 한번 비행을 시작하면 먹고 자는 모든 활동을 하늘에서 해결하며 "
                                 "지구를 몇 바퀴나 돌 정도로 놀라운 비행력을 자랑합니다.",
            "typeDescriptoon": "당신은 알바트로스처럼 드넓은 바다 위를 끊임없이 가장 빠르게 날아다니는 사람입니다. 짧은 여가 시간도 허투루 쓰지 않고, 활동적인 취미와 자기계발에 적극적으로 투자합니다. 헬스, 러닝, 등산 등과 같은 활동적인 취미를 통해 땀을 흘리고 성취감을 얻는 것을 즐깁니다. 당신에게 휴식은 잠시 숨 고르기일 뿐, 진정한 활력소는 도전하고 움직이는 순간에서 비롯됩니다.",
            "analSummary": [
                {
                    "subtitle": "⏰ 바쁘다 바빠 현대사회",
                    "content": "많이 바쁘신가요…? 이 유형은 주중과 주말 모두 모든 유형에 비해 가장 적은 여가시간을 보내고 있습니다.",
                    "metrics": [
                        {
                            "subtitle": "평일 일평균 여가시간",  # ⏰ 바쁘다 바빠 현대사회
                            "content": "전체 '평일 평균 여가시간'은 3.3시간이고, 이 유형은 '평일 평균 여가시간'을 2.1시간을 보냅니다. 또한 이 유형의 '평일 평균 여가시간'은 8가지 유형 중 가장 낮은 지표를 가지고 있습니다."
                        },
                        {
                            "subtitle": "주말 일평균 여가시간",  # ⏰ 바쁘다 바빠 현대사회
                            "content": "전체 '주말 평균 여가시간'은 6.0시간이고, 이 유형은 '주말 평균 여가시간'을 3.6시간을 보냅니다. 또한 이 유형의 '주말 평균 여가시간'은 8가지 유형 중 가장 낮은 지표를 가지고 있습니다."
                        }
                    ]
                },
                {
                    "subtitle": "내 건강은 내가",
                    "content": "내 건강은 내가 챙긴다! 이 유형은 여가시간을 사용함에 있어 [건강 관리]를 위한 목적이 가장 중요합니다.",
                    "metrics": [
                        {
                            "subtitle": "여가시간 사용 목적",  # 🩺 내 건강은 내가
                            "content": "이 유형 내에서 ‘여가시간 사용 목적 1순위’ 중 [건강 관리] 선택 비율이 27.1%로 가장 높았고, 모든 유형의 [건강 관리] 선택 비율은 12.7%로 다른 유형보다 [건강 관리]를 선택한 비율이 2.1배 수준입니다. 또한 다른 지표들의 수치는 비슷하기 때문에 무엇보다 [건강 관리]를 최우선으로 여기는 사람입니다."
                        }
                    ]
                },
                {
                    "subtitle": "🎾핸들이 고장난 알바트로스",
                    "content": "이 유형은 ‘여가시간 중 취미 사용 비율’이 모든 유형 중 가장 높고, ‘여가시간 중 휴식·오락 사용 비율’이 모든 유형 중 가장 낮습니다. 또한 ‘관심 여가활동’ 설문에서 [스포츠/운동]에 가장 높은 관심을 가지고 있기 때문에 여가시간 중 휴식보다 [스포츠/운동]의 취미를 추구하는 유형입니다.",
                    "metrics": [
                        {
                            "subtitle": "여가시간 중 취미 사용 비율",  # 🎾 스포츠·취미 우선형
                            "content": "이 유형은 ‘여가시간 중 취미 사용 비율’의 평균값이 40.8%으로 모든 유형의 평균 ‘여가시간 중 취미 사용 비율’인 18.9%의 2.2배 수준입니다."
                        },
                        {
                            "subtitle": "여가시간 중 휴식·오락 사용 비율",  # 🎾 스포츠·취미 우선형
                            "content": "이 유형은 ‘여가시간 중 휴식·오락 사용 비율’의 평균값이 13.3%으로 모든 유형의 평균 ‘여가시간 중 휴식·오락 사용 비율’인 42.2%의 0.3배 수준입니다."
                        },
                        {
                            "subtitle": "관심 여가활동 1순위",  # 🎾 스포츠·취미 우선형
                            "content": "이 유형의 ‘관심 여가활동 1순위’로 [스포츠/운동] 선택 비율은 35.8%, 2순위는 [없음]선택 비율이 28.7%이고 [스포츠/운동] 선택 비율은 18.8%, 3순위부터는 50%이상이 [없음]을 선택했습니다."
                        }
                    ]
                }
            ],
            "culsterInfo": [
                {"cNumber": "3"},
                {"type": "끈기의 비상가"}
            ],
            "interesting": ["분초사회", "효율끝판왕", "열정부자", "헬시플레저"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/albatross.png"
        }
    elif cluster == 4:
        resultvalue = {
            "animal": "푸른바다거북",
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
            "analSummary": [
                {
                    "subtitle": "💤 휴식 우선 전략",
                    "content": "[마음의 안정·휴식]을 우선시하는 유형입니다! 이 유형에서는 ‘여가시간 사용 목적 1,2순위’에서 [마음의 안정·휴식]이 비율이 가장 높았습니다!",
                    "metrics": [
                        {
                            "subtitle": "여가시간 사용 목적 1순위 비율",  # 💤 휴식 우선 전략
                            "content": "[마음의 안정·휴식을 위해]가 31.4%로 전체 유형 평균인 22.1%보다 1.4배 수준입니다."
                        },
                        {
                            "subtitle": "여가시간 사용 목적 2순위 비율",  # 💤 휴식 우선전략
                            "content": "[마음의 안정·휴식을 위해]가 27.8%로 전체 유형 평균인 21.7%보다 1.3배 수준입니다."
                        }
                    ]
                },
                {
                    "subtitle": "🌙 단독 몰입",
                    "content": "유형 중에서 ‘여가시간 중 대인관계·교제 사용 비율’이 평균적으로 가장 낮음과 동시에 ‘휴식/오락 비율’의 평균값이 전체 8개의 유형 중 가장 높습니다. 또한 ‘관심여가 활동’에서  [일상/휴식]에 치중되어 있어 혼자만의  시간을 즐기는 유형입니다!",
                    "metrics": [
                        {
                            "subtitle": "여가시간 중 대인관계·교제 사용 비율",  # 🌙 단독 몰입
                            "content": "이 유형에서는 평균값이 8.3%로 전체 유형 평균인 23%보다 0.4배 수준이며, 전체 유형 중 가장 낮습니다."
                        },
                        {
                            "subtitle": "여가시간 중 휴식·오락 사용 비율",  # 🌙 단독 몰입
                            "content": "이 유형에서는 평균값이 84%로 전체 유형 평균인 42%보다 2배 수준이며, 전체 유형 중 가장 높습니다."
                        },
                        {
                            "subtitle": "관심여가 활동",  # 🌙 단독 몰입
                            "content": "'관심여가 활동 1순위'에서 [일상/휴식] 비율이 37.2%로 전체 유형의 평균 20.9%와 비교대비 1.8배 수준이며, 대부분의 활동 카테고리에서 [없음]응답 비율이 월등히 높게 나타납니다."
                        }
                    ]
                }
            ],
            "culsterInfo": [
                {"cNumber": "4"},
                {"type": "슬로라이퍼"}
            ],
            "interesting": ["조용한 몰입", "감성루틴", "휴식이 최고"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/greenseaturtle.png"
        }
    elif cluster == 5:
        resultvalue = {
            "animal": "돌고래",
            "animalName": "당신의 동물은 돌고래입니다.",
            "animalType": "에너자이저 돌고래",
            "description": "당신은 에너지와 회복의 리듬을 아는 사람입니다. 야외활동을 우선시하되, 미디어·일상 휴식으로 컨디션을 영리하게 관리합니다.",
            "animalDescription": "돌고래는 거대한 무리를 이루어 복잡한 사회를 형성하며, "
                                 "무리별로 '유행가'를 공유할 만큼 뛰어난 지능과 문화적 유연성을 지녔습니다. "
                                 "이들은 배가 만드는 물살을 효율적으로 이용하여 최소한의 에너지로 "
                                 "먼 거리를 이동하는 지혜로움도 갖추고 있습니다. "
                                 "또한, 단체로 파도를 타며 서핑을 즐기거나 해파리로 공놀이를 하는 등 유희적인 행동을 자주 보여줍니다.",
            "typeDescriptoon": "당신은 돌고래처럼 움직임을 핵심으로 즐거움을 만듭니다. 스포츠·야외 활동에서 톤을 끌어올리고, 미디어/콘텐츠로 사이클을 부드럽게 이어갑니다.여기서 잠수와 호흡을 반복하듯 에너지를 효율적으로 관리하며 페이스 조절도 합니다. 또한 대인관계·교제 비율이 높아 활동적인 여가에서 함께할수록 시너지가 커집니다.",
            "analSummary": [
                {
                    "subtitle": "✨ 자기만족·즐거움 중심",
                    "content": "이 유형은 '여가시간 사용 목적'에서 [자기만족·즐거움]이 우선시되는 유형입니다! 하지만 그 [자기만족·즐거움]을 [마음의 안정·휴식]으로 충족시킬수도 있습니다!",
                    "metrics": [
                        {
                            "subtitle": "여가시간 사용 목적 1순위",  # ✨ 자기만족·즐거움 중심
                            "content": " [자기만족·즐거움을 위해]가 20.9%로 가장 높으며, 전체 유형 평균이 18.5%로 전체 유형 평균과 비교했을 때 1.1배 수준입니다."
                        },
                        {
                            "subtitle": "여가시간 사용 목적 2순위",  # ✨ 자기만족·즐거움 중심
                            "content": " [자기만족·즐거움을 위해]와 [마음의 안정·휴식을 위해]의 수치가 20.4%로 가장 높습니다. 각각 전체 유형 평균은 17.0%와 21.9%이며, [자기만족·즐거움을 위해]는 전체 평균보다 1.2배 수준이며, [마음의 안정·휴식을 위해]는 전체 평균보다 0.9배 수준입니다."
                        }
                    ]
                },
                {
                    "subtitle": "🏃 상호 보완적 활동",
                    "content": "'관심 여가활동 1~5순위'에서 [스포츠/운동]과[여행/야외]가 일관된 상위권을 유지했습니다. '평일/주말 평균 여가시간'을 고려했을 때 바쁜 일상 속에서도 활동적인 여가로 활기찬 에너지를 얻고, 콘텐츠 시청과 같은 활동으로 편안하게 휴식하며 삶의 균형을 능숙하게 조율하는 사람입니다.",
                    "metrics": [
                        {
                            "subtitle": "관심 여가활동 순위에서 [스포츠/운동]+[여행/야외]의 가중치 비율 ",  # 🏃 상호 보완적 활동
                            "content": "관심 여가활동 순위에서  [스포츠/운동]+[여행/야외]의 가중치의 값은 35.4%로 전체 유형 평균의 [스포츠/운동]+[여행/야외] 가중치 비율인 31.0%와 비교했을 때 1.1배 수준입니다."
                        },
                        {
                            "subtitle": "평일 일평균 여가시간",  # 🏃 상호 보완적 활동
                            "content": "'평일 일평균 여가시간'은 2.8시간으로 전체 유형 평균인 3.2시간보다 -0.4시간 낮았습니다."
                        },
                        {
                            "subtitle": "주말 일평균 여가시간",  # 🏃 상호 보완적 활동
                            "content": "'주말 일평균 여가시간'은 5.5시간으로 전체 유형 평균인 6.0시간보다 -0.5시간 낮았습니다."
                        }
                    ]
                }
            ],
            "culsterInfo": [
                {"cNumber": "5"},
                {"type": "에너자이저"}
            ],
            "interesting": ["밸런스", "운동과회복", "페이스메이커", "지속력", "똑똑한휴식"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/dolphin.png"
        }
    elif cluster == 6:
        resultvalue = {
            "animal": "펭귄",
            "animalName": "당신의 동물은 펭귄입니다.",
            "animalType": "균형의 수호자 펭귄",
            "description": "당신은 여유롭고 주도적으로 삶을 설계하는 사람입니다. 휴식·자기계발·관계를 균형 있게 조율하며, 주말을 활용하여 콘텐츠와 여행으로 마음을 채우고 삶을 풍요롭게 가꿉니다.",
            "animalDescription": "펭귄은 영하 수십 도에 달하는 극한의 추위를 견뎌내기 위해 "
                                 "서로의 몸을 밀착시켜 열 손실을 최소화하는 '허들링' 전술을 사용합니다. "
                                 "이들은 천적의 위협이 도사리는 혹독한 환경 속에서도, "
                                 "가시 돋친 혀와 강력한 턱을 활용하여 먹이를 현명하게 사냥해냅니다. "
                                 "또한, 돌멩이로 보금자리를 짓고 그 위에 돌멩이를 쌓아 올리는 행위를 통해 서로에 대한 애정을 확인하기도 합니다.",
            "typeDescriptoon": "당신은 펭귄처럼 사회생활이라는 육지에서 능숙하게 활동하면서도, 자신만의 바다(여가) 속에서는 유유히 헤엄치며 완벽한 균형을 이루는 사람입니다. 일에만 매달리지 않고 충분한 여가 시간을 확보하며, 휴식과 자기계발, 관계를 조화롭게 가꿔 삶을 풍요롭게 만듭니다. 휴식을 통해 에너지를 충전하고 이를 자기만족과 긍정적인 관계로 전환해 주변에도 활력을 전하는, 조화롭고 성숙한 삶의 주인공입니다.",
            "analSummary": [
                {
                    "subtitle": "⚖️ 균형 잡힌 생활자",
                    "content": "이 유형은 전반적인 여가 활동 분포에서 큰 차이 없이 안정적인 패턴을 보이며, 특히 자기계발과 대인 관계 활동을 조금 더 활발하게 즐기는 특징이 있습니다. 큰 치우침 없이 안정적인 여가 활용 패턴을 보여 ‘균형 잡힌 생활자’라 할 수 있습니다.",
                    "metrics": [
                        {
                            "subtitle": "여가 시간 분포의 균형",
                            "content": "이 유형의 평일·주말 여가시간은 전체 평균보다 다소 길지만, 휴식·취미·자기계발·교제 비율은 전체와 큰 차이가 없어 전체 분포와 유사한 패턴을 보입니다. 특정 영역에 치우치지 않고 고르게 분포해 ‘균형 잡힌 생활자’의 특성을 드러냅니다."
                        }
                    ]
                },
                {
                    "subtitle": "🛋️ 함께 쉬는 사람들",
                    "content": "이 유형은 사람과의 교제 속에서 일상의 휴식을 찾는 타입입니다. 사교·가족 목적과 일상/휴식 목적 비율이 전체 평균보다 두드러지게 높아, 사람과 함께하는 시간이 곧 휴식이자 여가의 핵심이 됩니다.",
                    "metrics": [
                        {
                            "subtitle": "여가 활동: 사교·가족 및 휴식",
                            "content": "이 유형은 여가 목적에서 ‘사교·가족’을 선택한 비율이 약 18.4%로 전체 유형(약 9%)의 두 배에 달하며, ‘일상·휴식’ 역시 36.4%로 전체 유형(약 21%) 대비 15%p 이상 높습니다. 이는 가족·지인과 함께하는 시간을 중시하고, 안정과 휴식을 통해 여가를 보내는 특징을 보여줍니다."
                        }
                    ]
                }
            ],
            "culsterInfo": [
                {"cNumber": "6"},
                {"type": "균형의 수호자"}
            ],
            "interesting": ["워라밸마스터", "균형의미학", "스마트투자", "긍정적리더", "성장지향"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/penguin.png"
        }
    else:
        resultvalue = {
            "animal": "해파리",
            "animalName": "당신의 동물은 해파리입니다.",
            "animalType": "유유자적 해파리",
            "description": "당신은 자신만의 평화를 지키며 여유롭게 삶을 보내는 사람입니다. 일상 속에서도 마음의 안정을 우선시하고, 스스로의 방식으로 스트레스를 풀어갑니다.",
            "animalDescription": "해파리는 플랑크톤의 일종으로, 스스로 헤엄칠 수 없어 바다의 흐름에 몸을 맡긴 채 이동합니다. "
                                 "뇌 없이 단순한 신경망만으로 세상을 감지하는 해파리는 주변 환경을 온몸으로 받아들이며 살아갑니다. "
                                 "또한, 신체 일부가 손상되어도 빠른 속도로 재생해내는 탁월한 회복력을 가지고 있습니다.",
            "typeDescriptoon": "당신은 해파리처럼 삶의 빠른 물살을 거스르지 않고, 잔잔한 바다를 여유롭게 유영하며 방해 받지 않는 편안함으로 에너지를 채워나가는 사람입니다. 현재의 삶에 지치더라도 최선을 다해 회복을 추구하는 현명함을 가지고 있군요. 당신에게 여가란 최소한의 에너지를 소모하며 ‘내일’을 맞이할 힘을 얻는 생존 모드이자, 반드시 찾아올 기회로부터 멋진 도약을 위해 몸을 움츠려 힘을 비축하는 시간입니다.",
            "analSummary": [
                {
                    "subtitle": "⏳ 시간 부자",
                    "content": "당신은 모든 유형 중 가장 많은 여가 시간을 보내는군요!",
                    "metrics": [
                        {
                            "subtitle": "평균 일평균 여가 시간",  # ⏳ 시간 부자
                            "content": "여가 시간이 평일 평균 5.6시간, 주말 평균 7.5시간으로, 전체 유형 중 가장 많은 여가 시간을 가집니다. 또한 전체 유형의 평일 일평균 여가 시간 3.2시간, 주말 일평균 여가시간 6.0시간보다 각각 1.8배, 1.3배 수준입니다. "
                        }
                    ]
                },
                {
                    "subtitle": "😴 휴식이 보약",
                    "content": "당신은 여가 시간의 대부분을 휴식과 오락에 사용하며, 활동적인 취미나 자기계발보다는 [미디어/콘텐츠]나 [일상/휴식]과 같이 조용하고 편안한 활동을 선호하는 사람입니다. 당신에게 여가란 일상에서 벗어나 에너지를 회복하는 시간입니다!",
                    "metrics": [
                        {
                            "subtitle": "여가시간 사용 목적 ",  # 😴 휴식이 보약
                            "content": " '여가시간 사용 목적 1순위'가 [마음의 안정·휴식]선택 비율이 49.1% 차지했으며, 전체 유형 평균 비율인 25.1% 대비 1.9배 수준입니다. 또한 '여가시간 사용목적 2순위'에서는 [스트레소 해소]선택 비율이 21.8%를 차지했으며, 전체 유형 평균 비율 13% 대비 1.7배 수준입니다. "
                        }
                    ]
                },
                {
                    "subtitle": "🪫 충전이 필요해!",
                    "content": "당신은 '여가 시간 중 취미와 자기계발의 비율'이 전체 유형 중에서 가장 낮지만 당신은 단순히 비활동적인 것이 아니라, 현재의 삶에서 가장 필요한 휴식에 집중하는 현명한 사람들일 수 있습니다.",
                    "metrics": [
                        {
                            "subtitle": "관심 여가 활동 ",  # 🪫 충전이 필요해!
                            "content": "이 유형의 '관심 여가 활동 1순위'는 43.6%를 차지한 [미디어/콘텐츠]이며 전체 유형의 평균인 22%보다 2.0배 수준입니다. '관심 여가 활동 2순위'부터는 36% 이상이 [일상/휴식] 항목을 선택했으며 전체 유형의 평균인 19.8%대비 1.8배 수준입니다."
                        }
                    ]
                }
            ],
            "culsterInfo": [
                {"cNumber": "7"},
                {"type": "유유자적"}
            ],
            "interesting": ["슬기로운 집콕생활", "소확행", "내 방이 최고"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/jellyfish.png"
        }
    return resultvalue


@router.post("/questions")
async def clusteranalyze(body: KmIn):
    x = np.array([[body.householdIncome, body.leisurePurpose, body.leisurePurpose2,
                   body.weekdayAvgLeisureTime, body.weekendAvgLeisureTime, body.restRecreationRate,
                   body.hobbyRate, body.selfImprovementRate, body.socialRelationshipRate,
                   body.leisureActivity1, body.leisureActivity2, body.leisureActivity3,
                   body.leisureActivity4, body.leisureActivity5]],
                 dtype=int)  # 설문조사 값 받아오기
    cluster = int(pipe.predict(x)[0])  # 모델 실행 후 클러스터 값 저장
    analyze = result_classify(cluster)
    return analyze
