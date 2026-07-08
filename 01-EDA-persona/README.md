"""
Nemotron-Personas-Korea EDA + 최애 주민 뽑기
한양대 ERICA 캡스톤 (롯데이노베이트 협업) 프로젝트

데이터 출처: https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea (CC BY 4.0)
"""

# ===== 0. 설치 (최초 1회) =====
# pip install datasets pandas matplotlib seaborn scikit-learn

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42

# ===== 1. 한글 폰트 설정 (Colab 기준) =====
# !apt-get -qq install fonts-nanum > /dev/null
# import matplotlib.font_manager as fm
# fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


# ===== 2. 데이터 로드 (스트리밍) =====
# 100만 행 전체를 한 번에 pandas로 올리면 메모리 부담이 크므로,
# 컬럼 구조 확인은 스트리밍 1건, EDA는 5만 건 샘플, 지역 필터링은 전체 순회로 나눠서 처리한다.
dataset = load_dataset("nvidia/Nemotron-Personas-Korea", streaming=True)

sample = next(iter(dataset["train"]))
print("컬럼 목록:", list(sample.keys()))
print("필드 수:", len(sample.keys()))


# ===== 3. EDA용 샘플 추출 (5만 건) =====
subset = dataset["train"].take(50000)
df = pd.DataFrame(subset)

print("\nshape:", df.shape)
print("\n[결측치]")
print(df.isnull().sum())
print("\n[성별]")
print(df["sex"].value_counts())
print("\n[나이 통계]")
print(df["age"].describe())
print("\n[시도별 상위 10]")
print(df["province"].value_counts().head(10))
print("\n[학력 분포]")
print(df["education_level"].value_counts())
print("\n[혼인 상태]")
print(df["marital_status"].value_counts())
print("\n[직업 상위 20]")
print(df["occupation"].value_counts().head(20))


# ===== 4. 시각화 (4분할) =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

df["age"].hist(bins=30, ax=axes[0, 0])
axes[0, 0].set_title("연령 분포")

df["province"].value_counts().head(10).plot(kind="barh", ax=axes[0, 1])
axes[0, 1].set_title("시도별 인원 (상위 10)")
axes[0, 1].invert_yaxis()

df["education_level"].value_counts().plot(kind="bar", ax=axes[1, 0])
axes[1, 0].set_title("학력 분포")
axes[1, 0].tick_params(axis="x", rotation=45)

df["marital_status"].value_counts().plot(kind="pie", ax=axes[1, 1], autopct="%1.1f%%")
axes[1, 1].set_title("혼인 상태")
axes[1, 1].set_ylabel("")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150)
plt.show()

# 지역별 학력 크로스탭 (시도 단위)
edu_by_province = pd.crosstab(df["province"], df["education_level"], normalize="index")
print("\n[시도별 학력 분포]")
print(edu_by_province)


# ===== 5. 지역 필터링: 고향(고창군) 주민 추출 =====
# 5만 건 샘플에는 특정 시군구 인원이 거의 안 잡히므로, 전체 스트리밍을 순회해서 필터링한다.
gochang_only = [row for row in dataset["train"] if "고창" in row["district"]]
gochang_df = pd.DataFrame(gochang_only)
print(f"\n고창군 표본 수: {len(gochang_df)}")
print(gochang_df["occupation"].value_counts().head(10))


# ===== 6. 최애 주민 뽑기 =====
def print_persona(row: pd.Series) -> None:
    fields = [
        "uuid", "sex", "age", "occupation", "district", "province",
        "education_level", "marital_status", "family_type",
        "persona", "professional_persona", "sports_persona", "arts_persona",
        "travel_persona", "culinary_persona", "family_persona",
    ]
    for f in fields:
        print(f"[{f}]\n{row[f]}\n")


# 방법 A: 페르소나 텍스트 풍부함 점수화 → 상위 10명 추출 (텍스트 길이 + 어휘 다양성)
from sklearn.preprocessing import MinMaxScaler

persona_cols = [
    "professional_persona", "sports_persona", "arts_persona",
    "travel_persona", "culinary_persona", "family_persona", "persona",
]
gochang_df["full_persona"] = gochang_df[persona_cols].agg(" ".join, axis=1)
gochang_df["text_length"] = gochang_df["full_persona"].apply(len)


def lexical_diversity(text: str) -> float:
    words = text.split()
    return len(set(words)) / len(words) if words else 0


gochang_df["lexical_diversity"] = gochang_df["full_persona"].apply(lexical_diversity)

scaler = MinMaxScaler()
gochang_df[["len_norm", "div_norm"]] = scaler.fit_transform(
    gochang_df[["text_length", "lexical_diversity"]]
)
gochang_df["interest_score"] = gochang_df["len_norm"] * 0.5 + gochang_df["div_norm"] * 0.5

top10 = gochang_df.sort_values("interest_score", ascending=False).head(10)
print("\n===== 고창군 개성있는 페르소나 Top 10 =====")
for _, row in top10.iterrows():
    print(f"[{row['uuid'][:8]}] {row['age']}세 / {row['sex']} / {row['occupation']}")
    print(row["persona"])
    print("-" * 60)

# 방법 B: 최종 선정 (직접 검토 후 확정)
# Top 10 중 캐릭터 대비가 가장 뚜렷하고 고창의 지역 산업 특성(농수산물)을 반영하는 인물로 최종 선정
FINAL_UUID_PREFIX = "9aee99c0"  # 정진식 씨
my_favorite = gochang_df[gochang_df["uuid"].str.startswith(FINAL_UUID_PREFIX)].iloc[0]

print("\n===== 최종 최애 주민 =====")
print_persona(my_favorite)
