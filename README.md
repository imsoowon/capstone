# Nemotron-Personas-Korea EDA: 나의 최애 주민 뽑기

> NVIDIA가 공개한 한국어 합성 페르소나 데이터셋(`nvidia/Nemotron-Personas-Korea`)을 탐색적으로 분석하고,
> 그 안에서 "최애 주민" 한 명을 뽑아보는 프로젝트 기록입니다.

## 데이터셋 소개

- **규모**: 100만 레코드, 700만 개 페르소나 텍스트, 총 17억 토큰
- **필드 26개**: 페르소나 텍스트 7종 + 속성 6종 + 인구통계·지리 12종 + UUID
- **커버리지**: 17개 시도, 252개 시군구, 이름 20.9만 개 고유 조합
- **생성 방식**: NVIDIA NeMo Data Designer (PGM + LLM 기반 합성), KOSIS·대법원·건강보험공단 등 실제 통계를 시드로 사용
- **라이선스**: CC BY 4.0 (상업적 사용 가능)

---

## STEP 1. 환경 준비 & 데이터 로드

대용량(100만 행) 데이터셋이라 전체 다운로드 대신 `streaming=True`로 필요한 만큼만 불러왔습니다.

```python
!pip install datasets pandas matplotlib seaborn -q

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Colab 기준 - 나눔고딕 별도 설치 필요)
!apt-get -qq install fonts-nanum > /dev/null
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 스트리밍 모드로 데이터 로드
dataset = load_dataset("nvidia/Nemotron-Personas-Korea", streaming=True)

# 컬럼 구조 확인 (1건만 확인 - 즉시 실행)
sample = next(iter(dataset["train"]))
print(list(sample.keys()))
```

### 확인된 실제 26개 필드

```
['uuid', 'professional_persona', 'sports_persona', 'arts_persona', 'travel_persona',
 'culinary_persona', 'family_persona', 'persona', 'cultural_background',
 'skills_and_expertise', 'skills_and_expertise_list', 'hobbies_and_interests',
 'hobbies_and_interests_list', 'career_goals_and_ambitions', 'sex', 'age',
 'marital_status', 'military_status', 'family_type', 'housing_type',
 'education_level', 'bachelors_field', 'occupation', 'district', 'province', 'country']
```

- **지리 필드**: `province`(시/도) + `district`(시/군/구)로 분리
- **페르소나 텍스트 7종**: professional / sports / arts / travel / culinary / family / persona(종합요약)
- **인구통계**: sex, age, marital_status, military_status, family_type, housing_type, education_level, bachelors_field, occupation

---

## STEP 2. EDA (탐색적 데이터 분석)

전체 분포 파악을 위해 스트리밍으로 5만 건 샘플을 추출해 시각화했습니다.

```python
subset = dataset["train"].take(50000)
df = pd.DataFrame(subset)

# 1. 연령 분포
sns.histplot(df['age'], bins=30)
plt.title('연령 분포')
plt.show()

# 2. 시/도별 인원수
df['province'].value_counts().plot(kind='barh')
plt.title('시/도별 인원수')
plt.show()

# 3. 직업 분포 Top 15
df['occupation'].value_counts().head(15).plot(kind='barh')
plt.title('직업 분포 Top 15')
plt.show()

# 4. 성별 x 가구유형
pd.crosstab(df['sex'], df['family_type']).plot(kind='bar', stacked=True)
plt.title('성별 x 가구유형')
plt.show()

# 5. 시/도별 학력 비교
edu_by_province = pd.crosstab(df['province'], df['education_level'], normalize='index')
edu_by_province
```

### EDA 인사이트

`province`별 `education_level` 크로스탭 결과, 몇 가지 뚜렷한 패턴이 확인됐습니다.

- **세종**이 대학원 비율 12.8%로 압도적 1위, 대졸까지 합치면 학력 수준 최상위
- **전라남·전북**은 무학·초등학교 비율이 다른 지역 대비 눈에 띄게 높음 (고령 인구 비중 반영으로 추정)
- **서울**은 대졸 34.9%로 높지만 대학원(8.7%)은 세종보다 낮음

---

## STEP 3. 고향 지역 필터링 (전북 고창군)

252개 시군구 중 특정 지역(고창군) 표본은 5만 건 샘플로는 거의 안 잡히기 때문에, 스트리밍 전체 순회 방식으로 필터링했습니다.

```python
gochang_only = [row for row in dataset["train"] if '고창' in row['district']]
gochang_df = pd.DataFrame(gochang_only)
print(gochang_df.shape)  # (1015, 26)
```

**결과**: 전북 고창군 표본 **1,015명** 확보 (전체 100만 건 중 약 0.1%, 실제 고창군 인구 비중과 유사한 수준)

---

## STEP 4. 최애 주민 선정 과정

**선정 기준**: 페르소나 텍스트가 가장 개성 있고 재밌는 사람

1,015명 전체를 다 읽을 수 없으므로, 텍스트 지표로 후보를 1차로 추린 뒤 직접 검토했습니다.

```python
from sklearn.preprocessing import MinMaxScaler

persona_cols = ['professional_persona','sports_persona','arts_persona',
                 'travel_persona','culinary_persona','family_persona','persona']
gochang_df['full_persona'] = gochang_df[persona_cols].agg(' '.join, axis=1)

# 텍스트 길이
gochang_df['text_length'] = gochang_df['full_persona'].apply(len)

# 어휘 다양성
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0

gochang_df['lexical_diversity'] = gochang_df['full_persona'].apply(lexical_diversity)

scaler = MinMaxScaler()
gochang_df[['len_norm','div_norm']] = scaler.fit_transform(
    gochang_df[['text_length','lexical_diversity']]
)
gochang_df['interest_score'] = gochang_df['len_norm']*0.5 + gochang_df['div_norm']*0.5

top10 = gochang_df.sort_values('interest_score', ascending=False).head(10)
```

### 최종 후보 3인 비교

| 이름 | 나이/성별 | 직업 | 특징 |
|---|---|---|---|
| 김희자 | 88세/여 | 화장품 판매원 | 고령에도 경제활동, 당뇨·혈압 안고도 소소한 행복 추구 |
| 배성복 | 90세/남 | 무직 | 서예·독서(정적) + 노래방(동적)의 상반된 취미 조합 |
| **정진식** | **53세/남** | **농수산물 중개인** | **호탕함 vs 섬세함의 뚜렷한 대비, 지역 산업 특성 반영** |

**최종 선정: 정진식 씨** — 캐릭터 대비가 가장 뚜렷하고, 직업 자체가 고창(농수산물 산지)이라는 지역색을 가장 직접적으로 반영하는 캐릭터라 "고창군을 대표하는 가상 주민"으로 적합하다고 판단.

---

## STEP 5. 최애 주민 카드

## 🏠 정진식 씨

| 항목 | 내용 |
|---|---|
| 나이/성별 | 53세 / 남자 |
| 지역 | 전북 고창군 |
| 직업 | 농수산물 중개인 및 경매사 |
| 학력 | 고등학교 |
| 가구유형 / 혼인상태 | 부모와 동거 / 미혼 |

### 한 줄 요약
> 고창의 흙과 바다를 닮아 호탕하고 현실적이며, 경매장의 카리스마와 마당 분재의 섬세함을 동시에 지닌 50대 미혼 남성.

### 직업 (professional_persona)
고창 농협 경매장에서 벼락같이 쏟아지는 호가 속에서도 최상품의 겉모양과 색깔을 단숨에 짚어내며, 특유의 쩌렁쩌렁한 성량으로 시장 분위기를 주도하는 베테랑 중개인. 상인들과의 팽팽한 기싸움 끝에 적절한 타협점을 찾아내면서도, 자신의 기준에 못 미치는 물건에는 가차 없이 냉정한 평가를 내리며 실리를 챙긴다.

### 취미/스포츠 (sports_persona)
주말이면 직접 운전해 고창읍내 외곽의 한적한 도로를 달리며 스트레스를 풀고, 집 마당에서 흙을 만지며 수국 가지를 치고 분재의 모양을 잡는 정적인 활동에서 깊은 안정을 찾는다. 친구들과의 야유회에서는 분위기 메이커를 자처하며 앞장서서 짐을 챙기고 일정을 짜는 등 에너지를 발산하는 활동에도 능숙하다.

### 예술 감각 (arts_persona)
정교하게 다듬어진 분재의 곡선미에서 예술적 쾌감을 느끼며, 마당의 수국이 계절에 따라 색을 바꾸는 과정을 세밀하게 관찰하고 기록하는 것에 소소한 기쁨을 느낀다. 화려한 전시회보다는 고창 갯벌의 풍경이나 보리밭의 물결 같은 자연이 주는 투박한 아름다움을 더 가치 있게 여긴다.

### 여행 (travel_persona)
고향 친구들과 함께 전북 인근의 산세가 수려한 곳이나 탁 트인 바다 풍경을 찾아다니며, 경치 좋은 정자 아래에서 옛 추억을 안주 삼아 술잔을 기울이는 여행을 즐긴다. 계획을 꼼꼼히 세우기보다는 발길 닿는 대로 움직이다가 우연히 발견한 숨은 명소에서 한참을 멍하니 풍경을 감상하곤 한다.

### 음식 (culinary_persona)
퇴근 후 고창읍내의 단골 갈빗집에서 지글지글 익어가는 고기에 소주 한 잔을 곁들이는 일상을 보내며, 가끔은 정읍까지 차를 몰아 정갈한 회 정식이 나오는 일식집에서 조용히 식사하는 시간을 갖는다. 배달 음식은 거의 쳐다보지 않고 직접 식당에 가서 주인장과 짧은 안부를 나누며 갓 만든 음식을 먹는 것을 당연하게 생각한다.

### 가족 (family_persona)
고창의 오래된 단독주택에서 연로하신 부모님을 정성껏 모시며 살아가고 있으며, 집안의 크고 작은 대소사를 도맡아 처리하는 든든한 외아들의 역할을 수행한다. 부모님께는 때로 무뚝뚝하고 고집스럽게 굴 때가 있지만, 시장 상인들이나 동네 어르신들에게는 깍듯한 예의를 갖추며 지역 공동체의 유대감을 중요하게 생각한다.

---

## 프로젝트 마무리

Nemotron-Personas-Korea 100만 건의 합성 데이터 중, 스트리밍을 통해 전체 다운로드 없이도 필요한 부분만 효율적으로 탐색할 수 있었고, 지역(고창군) 필터링으로 1,015명을 추려낸 뒤 텍스트 분석 + 직접 검토로 최종 1명을 선정했다. 단순한 데이터 필터링을 넘어, 합성 데이터셋이 얼마나 디테일하게 인물을 구성하는지(직업·취미·가족관계·식단까지) 체감한 것이 이번 분석의 수확이었다.

### 사용 기술
`Python` `pandas` `Hugging Face datasets` `matplotlib` `seaborn` `scikit-learn`

### 데이터 출처
[nvidia/Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea) (CC BY 4.0)
