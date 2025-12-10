from rec import Recommender
from knowledge import KnowledgeGraph

# 1) 추천 기사 리스트 가져오기
rec = Recommender(
    host="localhost",
    database="nvisiaDb",
    user="postgres",
    password="postgres1202",
    port=5432
)

selected_id = "spnews_101404"  
rec_list = rec.get_similar_articles(selected_id, k=10)

# 2) Knowledge Graph 생성 & 시각화
know = KnowledgeGraph(rec_list)
know.get_graph()

rec.close()
