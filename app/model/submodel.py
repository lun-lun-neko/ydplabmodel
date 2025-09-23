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
            "animalName": "당신의 동물은 범고래입니다",
            "animalType": "위풍당당 범고래",
            "description": "당신은 스스로 삶을 개척하며 끊임없이 성장하는 사람입니다. 건강 관리와 취미, 자기계발로 자신을 단련하고 삶의 질을 높여갑니다.",
            "animalDescription": "범고래는 바다의 최상위 포식자답게 무리를 지어 역할을 분담하고, "
                                 "주변 지형과 조류를 영리하게 활용하며, "
                                 "심지어 인공적으로 파도를 일으켜 먹잇감을 사냥하는 등, "
                                 "매우 지능적인 전략을 구사합니다. "
                                 "이들은 최대 시속 56km로 바다를 가르고 하루에 100km 이상을 "
                                 "거침없이 유영할 수 있는 빠르고 강력한 신체 능력을 자랑합니다.",
            "clusterDescription": "**- 강점**"
                                  ""
                                  "- **활동적 재충전:** ‘쉼’ 대신 ‘운동’으로 에너지를 채우는 유형. 여가 목적 1순위 **“건강 관리”**."
                                  "(**활동적 재충전:** ‘쉼’ 대신 ‘운동’으로 에너지를 채우는 유형입니다!"
                                  "이 유형의 몇%는 여가 목적 1순위로 “건강 관리”를 가장 ㅏㅁㄹ니 선택했습니다!)"
                                  "- **성장 지향적 여가:** 취미(33.42%)를 즐기면서도 자기계발(21.36%)을 놓치지 않음."
                                  "- **경계 없는 탐험:** 익숙함에 안주하지 않고 ‘스포츠’, ‘여행’ 등 새로운 도전을 즐김"
                                  ""
                                  "**- 핵심 지표 (근거)**"
                                  ""
                                  "- **활동 비율:** **휴식/오락**에 할애하는 비중(**42.83%**)이 전체 평균(16.76%)보다 **약 2.5배 이상 높습니다.** 반대로 취미, 자기계발, 사교 활동 비율은 평균보다 현저히 낮습니다."
                                  "- **주요 활동:** 'TV 시청' 비율이 평균 대비 압도적으로 높으며, 대부분의 활동 카테고리에서 '참여하는 활동이 없다'는 응답 비율이 월등히 높게 나타납니다."
                                  ""
                                  "**- 타입 소개**"
                                  ""
                                  "당신은 무리를 이끄는 범고래처럼 현재의 성공에 안주하지 않고, 끝없이 새로운 영역을 탐험하며 삶을 개척하는 리더입니다. 여가 시간을 단순한 휴식이 아닌 **건강 관리**와 **성장**을 위한 투자로 여기며, 평균보다 적은 시간임에도 불구하고 **스포츠와 자기계발** 같은 생산적인 활동으로 알차게 채웁니다. 이러한 삶의 태도는 당신이 주도권을 쥐고, 자신을 단련하며 균형 잡힌 최상의 삶을 만들고자 하는 강한 의지를 명확히 보여줍니다.",
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
            "clusterDescription": "**- 강점**"
                                  ""
                                  "- 🤝 **찐친파의 즐거움:** '가족·지인과 시간 보내기'가 여가 목적 2순위입니다. 다른 군집보다 사회적 관계를 중시합니다."
                                  "- ✨ **경험형 활동가:** 일상 속 작은 여행, 야외 활동으로 리프레시를 즐깁니다."
                                  "- 🏃‍♀️ **적당한 활동:** '스트레스 해소'와 '건강 관리'를 동시에 추구하는 유형입니다."
                                  ""
                                  "**- 핵심 지표 (근거)**"
                                  ""
                                  "- **여가 목적 1순위:** '스트레스 해소'(20.7%)가 가장 높습니다."
                                  "- **관심 레저활동 1순위:** '여행/야외활동'(24.6%)이 다른 군집보다 가장 높은 비율을 차지합니다."
                                  "- **취미·자기계발 참여율:** 낮고 '없음' 응답 비율이 높습니다."
                                  ""
                                  "**- 타입 소개**"
                                  ""
                                  "당신은 **왜가리처럼 조용한 물가에서 자신만의 리듬으로 하루를 살아가는** 사람입니다. 여가 시간이 길지 않아도 소중한 사람들과의 관계를 통해 활력을 얻습니다. 거창한 모험보다는 익숙한 사람들과 함께 떠나는 가벼운 여행, 맛있는 음식을 나누며 수다를 떠는 소소한 경험에서 삶의 만족을 찾습니다. 몸과 마음의 스트레스를 해소하는 것을 가장 중요하게 생각하며, 혼자만의 시간보다는 **함께하는 즐거움**을 통해 에너지를 재충전합니다.",
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
            "clusterDescription": "**- 강점**"
                                  ""
                                  "- 🧘 **마음 안정 지향:** 레저 목적 1·2순위 모두 **'마음의 안정·휴식'**을 최우선으로 둡니다."
                                  "- 💤 **초고효율 회복:** 휴식·오락 비중(57%)이 모든 군집 중 가장 높아, 충분한 휴식으로 컨디션을 관리합니다."
                                  "- 🧭 **선택과 집중:** 취미나 자기계발 활동을 줄이고 오롯이 '필요한 쉼'에 집중합니다."
                                  ""
                                  "**- 핵심 지표 (근거)**"
                                  ""
                                  "- **여가 목적 1순위:** '마음의 안정·휴식'(34.4%)이 압도적으로 높습니다."
                                  "- **여가 시간:** 주중·주말 여가 시간이 전체 평균보다 각각 0.5~1시간 길어, 충분한 휴식 시간을 확보합니다."
                                  "- **관심 레저활동 1순위:** '일상/휴식'(30.4%)과 '미디어/콘텐츠'(24.5%) 비율이 높습니다."
                                  ""
                                  "**- 타입 소개**"
                                  ""
                                  "당신은 **매너티처럼 느긋한 물살에 몸을 맡기고, 평온한 리듬 속에서 하루를 사는** 사람입니다. 바쁜 성취보다는 여유로운 휴식을 통해 스스로를 돌보고, 따뜻한 관계 속에서 삶의 만족을 찾습니다. 커피 한 잔을 음미하며 책을 읽거나, 공원 벤치에서 햇살을 즐기는 등 **일상 속 소소한 행복**을 발견하며 에너지를 충전합니다.",
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
            "clusterDescription":"**- 강점**"
                                 ""
                                 "- 📈 **성장형 마인드:** 휴식보다 배우고 움직이며 삶의 활력을 찾습니다."
                                 "- 💪 **갓생러의 표본:** 여가 시간의 대부분을 운동·자기계발에 투자하며 높은 밀도를 자랑합니다."
                                 "- 🥇 **성취감 중독:** 땀과 성취로 에너지를 얻고 만족감을 높입니다."
                                 ""
                                 "**- 핵심 지표 (근거)**"
                                 ""
                                 "- **여가 목적 1순위:** '건강 관리'(27.1%)가 모든 군집 중 압도적으로 높습니다."
                                 "- **관심 레저활동 1순위:** '스포츠/운동'(35.8%)에 대한 관심이 압도적입니다."
                                 "- **여가 시간:** 주중·주말 여가 시간이 모든 군집 중 가장 짧습니다."
                                 ""
                                 "**- 타입 소개**"
                                 ""
                                 "당신은 **알바트로스처럼 넓은 바다 위를 끊임없이 날아다니는** 사람입니다. 짧은 여가 시간도 허투루 쓰지 않고, 운동·취미·자기계발에 적극적으로 투자합니다. 헬스, 러닝, 등산 등 **도전적인 활동**을 통해 땀을 흘리며 성취감을 얻고, 이를 통해 삶의 활력을 얻습니다. 당신에게 휴식은 잠시 숨 고르기일 뿐, 진짜 에너지는 움직이고 도전하는 순간에서 비롯됩니다.",
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
            "clusterDescription": "**- 강점**"
                                  ""
                                  "- 💤 **초고효율 회복**: 휴식·오락 비중 평균 **83.59%** (전체 **42.23%**)"
                                  "- 🧭 **선택과 집중**: 불필요한 활동을 줄이고 ‘필요한 쉼’에 리소스 올인"
                                  "- 🧘 **마음 안정 지향**: 레저 목적 1·2순위 모두 **“마음의 안정·휴식”**을 최우선"
                                  ""
                                  "**- 핵심 지표 (근거)**"
                                  ""
                                  "- **레저목적 1순위:** 마음의 안정·휴식(31.4%) → 전체 평균 대비 **압도적 상위**"
                                  "- **관심 레저활동 1순위:** 일상/휴식(37.2%) → **일상 속 회복 루틴 최적화,** 전체 평균 대비 압도적 상위"
                                  ""
                                  "**- 타입 소개**"
                                  "당신은 **푸른바다거북**처럼 느긋한 호흡으로도 꾸준히 멀리 나아갑니다. 영화·전시·독서 같은 정적 몰입에서 에너지를 채우고 산책으로 하루의 리듬을 부드럽게 순환시키며, 관람과 창작을 오가며 취향을 차곡차곡 아카이빙합니다. 휴식을 도피가 아닌 **다음 몰입을 위한 현명한 재충전**으로 다루고, 과시보다 **깊이와 지속성**을 중시합니다.",
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
            "clusterDescription":"**- 강점**"
                                 ""
                                 "- 🏃 **액티브 몰입**: 관심 1·2순위 모두 **스포츠/운동 비중 높음** → 체력·활력 관리 강점"
                                 "- 🧭 **즐거움·관계의 균형**: ‘자기만족/즐거움’과 ‘가족·지인과 시간’ **동시에 상위**"
                                 "- 🌿 **야외 순환**: ‘여행/야외활동’이 **다수 순위**에서 상위 → 회복과 자극의 균형"
                                 "- 🎯 **다영역 성장**: 레저시간 중 **대인관계·취미·본인계발 비율**이 **전체 평균보다 높음**"
                                 ""
                                 "**- 핵심 지표 (근거)**"
                                 ""
                                 "- 레저목적 1순위: **4. 자기만족·즐거움(20.9%)**, **3. 가족·지인과 시간(20.7%)** → 둘 다 **전체 평균 대비 상위**"
                                 "- 레저목적 2순위: **1. 마음의 안정·휴식(20.4%)**, **4. 자기만족·즐거움(20.4%)** → **공동 1위**"
                                 "- 관심 레저활동 1순위: **스포츠/운동(25.0%)** → **전체 평균보다 확실히 높음**"
                                 "- 관심 레저활동 2순위: **스포츠/운동(21.9%)**, **미디어/콘텐츠(21.1%)**"
                                 "- 관심 레저활동 3·4순위: **여행/야외활동(20.4%, 22.4%)**가 **각각 1위**, **미디어/콘텐츠(19.7%, 17.7%)**가 **각각 2위**"
                                 "- 레저시간 구성: **대인관계·교제 / 취미 / 본인계발** 비율이 **전체 평균보다 높음**"
                                 ""
                                 "**- 타입 소개**"
                                 ""
                                 "당신은 **돌고래**처럼 유연한 추진력으로 즐거움과 관계를 동시에 살려 성장을 만들어냅니다. 운동과 야외 활동에서 활력을 끌어올리고, 미디어·콘텐츠로 집중을 보강하며, 취미·자기계발·사교가 균형 있게 순환합니다. 즐거움을 에너지로 바꾸고, 그 에너지로 다시 몰입을 확장하는 **지속 가능한 액티브 타입**입니다.",
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
            "clusterDescription":"**- 강점**"
                                 ""
                                 "- 💤 **안정 회복 1순위**: 레저 목적 **휴식 33%** → 다른 선택지 대비 **압도적 우위**, 전체 평균보다 확실히 높음"
                                 "- 🏃 **액티브 에너지**: 관심 1순위 **스포츠/운동 32.7%** → 전체 평균 대비 뚜렷한 상위"
                                 "- 🌿 **로우키 회복 루프**: 관심 **2~5순위 모두 ‘일상/휴식’**이 압도적 상위 → 일상에서 꾸준히 충전"
                                 "- 🤝 **소셜 강점**: 레저시간 중 **대인관계·교제 27.38%** (전체 **23.19%**) → 관계에서 에너지 얻는 타입"
                                 ""
                                 "**- 핵심 지표(근거)**"
                                 ""
                                 "- 레저목적 1순위: **1. 마음의 안정·휴식(33%)**"
                                 "- 레저목적 2순위: **3. 가족·지인과 시간(19.6%)**"
                                 "- 관심 레저활동 1순위: **스포츠/운동(32.7%)**"
                                 "- 관심 레저활동 2~5순위: **일상/휴식**이 **모두** 전체 평균보다 높음"
                                 "- 레저시간 구성: **대인관계·교제 27.38%** vs 전체 **23.19%**"
                                 ""
                                 "**- 타입 소개**"
                                 "당신은 **펭귄**처럼 서로 기대어 관계의 온기를 나누고, 안정적인 루틴 속에서 꾸준히 성장합니다. 여유로운 주말엔 **콘텐츠 큐레이션·자기계발**로 깊이를 쌓고, 가끔은 **라이트 아웃도어**로 공기를 환기합니다. 소비와 시간 사용은 계획적이며, 나에게 맞는 것만 선별해 삶의 기준을 섬세하게 업그레이드합니다. **고요 속에서 확장**하는 법을 알고, 차분한 일상 설계로 높은 만족도를 오래 유지합니다.",
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
            "clusterDescription":"**- 강점**"
                                 ""
                                 "- 💤 **초심안정 최우선**: 레저 목적 **휴식 49.1%** → 전체 평균 대비 **압도적 상위**"
                                 "- 📺 **저자극 깊은 몰입**: 관심 1순위 **미디어/콘텐츠 43.6%**, 2위 **일상/휴식 35.9%**"
                                 "- 🌿 **일상형 회복 루프**: 관심 **2~5순위 모두 ‘일상/휴식’**이 지속 상위"
                                 "- 🧯 **스트레스 케어**: 목적 2순위 **스트레스 해소 21.8%** → 회복 설계에 강점"
                                 "- 📈 **회복 지표 우수**: **휴식·오락 비중 75.93%** (전체 **42.23%**)"
                                 ""
                                 "**- 핵심 지표 (근거)**"
                                 ""
                                 "- 레저시간 사용 목적 1순위: **1. 마음의 안정·휴식(49.1%)**"
                                 "- 레저시간 사용 목적 2순위: **2. 스트레스 해소(21.8%)**"
                                 "- 관심 레저활동 1순위: **1. 미디어/콘텐츠(43.6%)**, **7. 일상/휴식(35.9%)**"
                                 "- 관심 레저활동 2~5순위: **7. 일상/휴식**이 모두 전체 평균보다 높음"
                                 "- 레저시간 구성: **rest_recreation_rate 75.93%** vs 전체 **42.23%**"
                                 ""
                                 "**- 타입 소개**"
                                 ""
                                 "당신은 **해파리**처럼 물결을 타며 에너지를 아끼고, 필요한 순간에 깊이 회복하는 타입입니다. 미디어·콘텐츠와 일상적 휴식을 길게·안정적으로 즐기며, 스트레스는 부드러운 루틴으로 낮추고 컨디션을 차분히 끌어올립니다. **저자극 몰입 → 회복 → 재몰입**의 순환이 탁월합니다.",
            "interesting": ["슬기로운 집콕생활", "소확행", "내 방이 최고", "시간 만수르"],
            "animalImageUrl": "https://lunlunneko-ydplab.hf.space/static/animals/jellyfish.svg"
        }
    return resultvalue


@router.post("/v1/questions")
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
