import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast

class RecommenderTest:

    def __init__(self, csv_path):
        """

        [작성 2025-12-01] 
        테스트 목적의 CSV 기반 추천 시스템.
        
        input: 사용자가 선택한 기사 id (click_id)
        output: 관련 기사 

        postgres 없이도 실행 가능하도록 작성하였음.(postgres 버전은 Recommender)     

        [수정 2025-12-01]
        issue: 입력 id를 기준으로 row를 탐색하다보니 시간이 매우 오래걸림.
        해결: idx를 부여하여 빠르게 탐색하도록 수정. (id_to_index 추가)

        실행 방법은 rec_test.ipynb 파일 참고.
        
        """

        df = pd.read_csv(csv_path)

        # object > array
        df["embedding"] = df["embedding"].apply(ast.literal_eval)
        embeddings = np.array(df["embedding"].to_list())

        id_to_index = {id_val: idx for idx, id_val in enumerate(df["id"])}

        self.df = df
        self.embeddings = embeddings
        self.id_to_index = id_to_index

    def get_similar_articles(self, click_id, k):
        """

        [작성 2025-12-01]

        click_id: 사용자가 클릭한 기사 id
        k: 추천할 기사의 개수

        click_id의 embedding값을 참고하여 코사인 유사도를 계산.

        반환: json 형식.

 

        """

        click_idx = self.id_to_index[click_id]
        click_vec = self.embeddings[click_idx]

        # 코사인 유사도 계산
        sims = self.embeddings @ click_vec 

        # 본인 제외
        sims[click_idx] = -1.0

        # 상위 k개 리턴
        top_idx = np.argsort(-sims)[:k]  

        results = []
        for idx in top_idx:
            row = self.df.iloc[idx]

            results.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "publish_date": str(row["publish_date"])[:10],
                    "title": row["title"],
                    "url": row["url"]
                }
            )

        return results