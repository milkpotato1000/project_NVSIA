import pandas as pd
from project_NVSIA.test_python_files.LLMtoDatabase import LLMtoDatabase

df = pd.read_csv("data/test_region_parsing.csv", encoding="cp949")

llm_db = LLMtoDatabase(
    host="localhost",
    database="nvisiaDb",
    user="postgres",
    password="postgres1202",
    port=5432,
)

for idx, row in df.iterrows():
    title = str(row["title"])
    contents = str(row["contents"])
    publish_date = str(row["publish_date"])
    url = str(row["url"])

    if llm_db.check_url(url):
        print(f"[중복] 이미 존재하는 기사입니다. 이어서 다음 기사를 분석합니다.")
        continue

    llm_output = llm_db.get_article_summary(title, contents, publish_date)

    if llm_output is None:
        print(f"[에러] 데이터가 누락되어 다음 행으로 넘어갑니다. {idx}")
        continue

    llm_db.insert_summary(llm_output, title, publish_date, url)

    print(f"[저장] 행 업로드 되었습니다. {idx}")

llm_db.close()
print("[종료] 모든 업로드가 완료되었습니다.")





