"""
피드백 ② [심화] — 페르소나 구매 품목 인터뷰 (전체 실행용)

week8 페이지의 '에이전트별 로그 CSV'(agent_log_characters.csv)를 입력으로 받아,
캐릭터별로 N명씩 LLM(Gemini 무료 티어, week6과 같은 모델 계열)에게 "편의점에서 무엇을 샀나"를 묻고
item_interviews.json(페이지가 읽는 형식)으로 저장한다. 현재 페이지에는 Claude가 답한 예비 표본 60명이 들어 있다.

실행 (Colab 등):
    pip install google-genai pandas
    export GEMINI_API_KEY=...
    python interview_items.py agent_log_characters.csv --per-character 30 --out item_interviews.json
    # 결과를 페이지에 바로 넣으려면 (index.html 안의 const ITEMS=... 를 교체)
    python interview_items.py agent_log_characters.csv --per-character 30 --html index.html
"""
import argparse, json, os, random, time

import pandas as pd

CATEGORIES = ["커피·차", "생수·탄산·이온음료", "주류", "간편식사", "과자·디저트", "숙취해소·건강", "생활·위생용품", "담배"]
PROMPT = """당신은 아래 페르소나입니다. {slot}시에 서울 홍대입구역 상권에 있다가 편의점에 들렀습니다.
무엇을 샀나요? 실제 통계에 맞추려 하지 말고 페르소나의 성향대로 답하세요.
품목 분류는 반드시 다음 중에서 1~3개 고르세요: {cats}
JSON 한 줄로만 답하세요: {{"items":[...], "detail":"구체 품목", "reason":"이유 한 줄"}}

[페르소나] {age}세 {sex}, 직업: {occupation}, 손님 유형: {character}
{persona}
취미: {hobbies}"""


def ask(client, model, row):
    msg = PROMPT.format(slot=row.time_slot, cats=", ".join(CATEGORIES), age=row.age, sex=row.sex, occupation=row.occupation,
                        character=row.character, persona=row.persona, hobbies=row.hobbies)
    for attempt in range(4):
        try:
            r = client.models.generate_content(model=model, contents=msg)
            t = r.text.strip().strip("`").removeprefix("json").strip()
            d = json.loads(t[t.index("{"):t.rindex("}") + 1])
            d["items"] = [x for x in d.get("items", []) if x in CATEGORIES][:3]
            if d["items"]:
                return d
        except Exception as e:  # 429 등은 잠시 쉬고 다시
            time.sleep(4 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_log")
    ap.add_argument("--per-character", type=int, default=30)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--out", default="item_interviews.json")
    ap.add_argument("--html", help="결과를 넣을 index.html (선택)")
    a = ap.parse_args()
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    df = pd.read_csv(a.agent_log, encoding="utf-8-sig").drop_duplicates("uuid")
    rng = random.Random(a.seed)
    answers = []
    for ch, g in df.groupby("character"):
        for row in g.sample(min(a.per_character, len(g)), random_state=rng.randint(0, 10**6)).itertuples():
            d = ask(client, a.model, row)
            if d:
                answers.append({"i": int(row.agent_index), **d})
            time.sleep(1.2)  # 무료 티어 속도 제한
        print(ch, "완료")
    meta = {"interviewer": a.model, "n": len(answers), "categories": CATEGORIES,
            "sampling": f"에이전트 로그에서 캐릭터별 {a.per_character}명 무작위(시드 {a.seed})",
            "question": PROMPT.split("[페르소나]")[0].strip(),
            "caution": "LLM 응답은 실제 구매 기록이 아니라 페르소나 설정에서 추론한 가설입니다."}
    json.dump({"meta": meta, "answers": answers}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장:", a.out, len(answers), "명")
    if a.html:
        import re
        page = open(a.html, encoding="utf-8").read()
        new = "const ITEMS=" + json.dumps({"meta": meta, "answers": answers}, ensure_ascii=False) + ";</script>"
        page, n = re.subn(r"const ITEMS=.*?;</script>", lambda m: new, page, count=1, flags=re.S)
        assert n == 1, "index.html 에서 const ITEMS 를 찾지 못함"
        open(a.html, "w", encoding="utf-8").write(page)
        print("페이지 갱신:", a.html)


if __name__ == "__main__":
    main()
