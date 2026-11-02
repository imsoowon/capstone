"""
서울시 상권분석서비스 원자료에서 홍대입구역(홍대) 상권의 '편의점' 실측값을 뽑아
market_facts.json 으로 저장한다. 보정 방식은 서연(ashlsylee/capstone 7_store_entry_simulation)과 같다.

입력 (서울 열린데이터광장, 2025년 파일 그대로):
- 서울시 상권분석서비스(추정매출-상권)  OA-15572  → 업종별 매출 금액·건수 (시간대·요일·성별·연령 분해)
- 서울시 상권분석서비스(점포-상권)      OA-15577  → 업종별 점포 수 (유사업종 점포 수 = 일반 + 프랜차이즈)

출력: market_facts.json  (index.html 의 MARKET 상수로 들어간다)

실행:
    python prepare_market_data.py <추정매출 CSV> <점포 CSV> [--out market_facts.json]
"""
import argparse
import json
import sys

import pandas as pd

DISTRICT_CODE = 3120103          # 홍대입구역(홍대) 발달상권
INDUSTRY = "편의점"              # 서울시 서비스 업종명 (CS300002)
QUARTERS = [20251, 20252, 20253, 20254]

TIMES = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]
DAYS = ["월", "화", "수", "목", "금", "토", "일"]
DAY_FULL = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
AGES = ["10대", "20대", "30대", "40대", "50대", "60대+"]


def read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    sys.exit(f"인코딩을 읽지 못함: {path}")


def col(df: pd.DataFrame, *candidates: str) -> str:
    """같은 값을 가리키는 컬럼명이 파일(국문/영문 API)마다 달라서 후보 중 있는 것을 고른다."""
    for c in candidates:
        if c in df.columns:
            return c
    sys.exit(f"컬럼을 찾지 못함: {candidates}\n실제 컬럼: {list(df.columns)}")


def time_cols(df: pd.DataFrame, kind: str) -> list:
    """시간대 매출 컬럼. 원본 표기가 '시간대_00~06_매출_건수' 또는 '시간대_건수~06_매출_건수'로 섞여 있다."""
    out = []
    for a, b in [("00", "06"), ("06", "11"), ("11", "14"), ("14", "17"), ("17", "21"), ("21", "24")]:
        names = [f"시간대_{a}~{b}_매출_{kind}", f"시간대_{kind}~{b}_매출_{kind}", f"시간대_{a}_{b}_매출_{kind}"]
        out.append(col(df, *names))
    return out


def pct(values) -> list:
    t = float(sum(values))
    return [round(v / t * 100, 2) if t else 0.0 for v in values]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sales_csv")
    ap.add_argument("stores_csv")
    ap.add_argument("--out", default="market_facts.json")
    args = ap.parse_args()

    sales, stores = read_csv(args.sales_csv), read_csv(args.stores_csv)

    # ---- 추정매출-상권
    s_code = col(sales, "상권_코드", "TRDAR_CD", "trdar_cd")
    s_q = col(sales, "기준_년분기_코드", "STDR_YYQU_CD", "stdr_yyqu_cd")
    s_ind = col(sales, "서비스_업종_코드_명", "SVC_INDUTY_CD_NM", "svc_induty_cd_nm")
    d = sales[(sales[s_code].astype(int) == DISTRICT_CODE) & (sales[s_ind] == INDUSTRY)
              & (sales[s_q].astype(int).isin(QUARTERS))]
    got = sorted(d[s_q].astype(int).unique())
    if got != QUARTERS:
        sys.exit(f"추정매출: 2025년 4개 분기가 모두 있어야 함 (있는 분기: {got})")
    s = d.sum(numeric_only=True)
    amount = int(s[col(sales, "당월_매출_금액", "THSMON_SELNG_AMT")])
    count = int(s[col(sales, "당월_매출_건수", "THSMON_SELNG_CO")])
    t_cnt = [s[c] for c in time_cols(sales, "건수")]
    t_amt = [s[c] for c in time_cols(sales, "금액")]
    day_cnt = [s[col(sales, f"{d_}_매출_건수")] for d_ in DAY_FULL]
    age_cnt = [s[col(sales, f"연령대_{a}_매출_건수")] for a in ["10", "20", "30", "40", "50", "60_이상"]]
    sex_cnt = [s[col(sales, "남성_매출_건수")], s[col(sales, "여성_매출_건수")]]

    # ---- 점포-상권
    t_code = col(stores, "상권_코드", "TRDAR_CD", "trdar_cd")
    t_q = col(stores, "기준_년분기_코드", "STDR_YYQU_CD", "stdr_yyqu_cd")
    t_ind = col(stores, "서비스_업종_코드_명", "SVC_INDUTY_CD_NM", "svc_induty_cd_nm")
    t_n = col(stores, "유사_업종_점포_수", "SIMILR_INDUTY_STOR_CO", "similr_induty_stor_co")
    t_f = col(stores, "프랜차이즈_점포_수", "FRC_STOR_CO", "frc_stor_co")
    st = stores[(stores[t_code].astype(int) == DISTRICT_CODE) & (stores[t_ind] == INDUSTRY)
                & (stores[t_q].astype(int).isin(QUARTERS))]
    if len(st) != 4:
        sys.exit(f"점포: 2025년 4개 분기 행이 있어야 함 (행 수 {len(st)})")
    n_stores = round(float(st[t_n].mean()), 1)       # 분기별 점포 수 평균 (서연 방식)

    adult = age_cnt[1:]                               # 페르소나가 20세 이상이라 20세 이상 기준 비중도 함께 둔다
    facts = {
        "district_code": DISTRICT_CODE,
        "district_name": "홍대입구역(홍대)",
        "industry": INDUSTRY,
        "period": "2025년 1~4분기 합계",
        "source": "서울시 상권분석서비스 추정매출-상권(OA-15572)·점포-상권(OA-15577), 2025년",
        "stores": n_stores,
        "stores_by_quarter": {str(int(r[t_q])): int(r[t_n]) for _, r in st.iterrows()},
        "franchise_by_quarter": {str(int(r[t_q])): int(r[t_f]) for _, r in st.iterrows()},
        "sales": amount,
        "transactions": count,
        "avg_ticket": round(amount / count),
        "sales_per_store": round(amount / n_stores),
        "tx_per_store": round(count / n_stores),
        "time": {t: {"tx": int(c), "amount": int(a), "pct": p, "ticket": round(a / c) if c else round(amount / count)}
                 for t, c, a, p in zip(TIMES, t_cnt, t_amt, pct(t_cnt))},
        "day_tx_pct": dict(zip(DAYS, pct(day_cnt))),
        "age_tx_pct_raw": dict(zip(AGES, pct(age_cnt))),
        "age_tx_pct": dict(zip(AGES[1:], pct(adult))),
        "sex_tx_pct": dict(zip(["남자", "여자"], pct(sex_cnt))),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: facts[k] for k in ["stores", "sales", "transactions", "avg_ticket", "sales_per_store", "tx_per_store"]},
                     ensure_ascii=False))
    print("시간대 결제 비중:", {t: v["pct"] for t, v in facts["time"].items()})


if __name__ == "__main__":
    main()
