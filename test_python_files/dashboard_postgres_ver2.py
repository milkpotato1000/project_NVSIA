import numpy as np
import pandas as pd
import ast

# 시각화
import streamlit as st
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# LLM
from LLMtoDatabase import LLMtoDatabase

# 추천, 지도
from rec import Recommender
from geocoder import Geocoder
import folium
import random

# postgres
import psycopg2
from psycopg2.extras import RealDictCursor

NOTES = """

[작성 2025-12-03]
dashboard.py 내용 수정. postgres 연동 목적.

동작 순서:
    1) 기사 원문 csv 입력
    2) LLMtoDatabase 통해 summary, embedding, categorizing 진행
    3) Recommender 통해 관련 기사 k개 추천
    4) Geocoder 통해 지도에 위치 표시

[수정 2025-12-03]
- url 하이퍼링크 적용.

[수정 2025-12-04]
geocoder 통해 사용자가 클릭한 기사의 event_loc을 지도에 표시(추천 기사의 event_loc은 표시하지 않음).
Recommender, Geocoder 에서 모두 활용할 selected_id 객체 생성하였음.

[수정 2025-12-08]
issue: 지도 출력 event 무한 호출.
solve: 

"""

# ========================= #
# DB 설정
# ========================= #
DB = dict(
    host="localhost",
    database="nvisiaDb",
    user="postgres",
    password="postgres1202",
    port=5432,
)

# ========================= #
# 공용 커넥터 / 헬퍼
# ========================= #
def get_psql_conn():
    """간단 쿼리용 psycopg2 커넥션 (글로벌 캐시 X, 매번 열고 닫기)"""
    conn = psycopg2.connect(
        host=DB["host"],
        database=DB["database"],
        user=DB["user"],
        password=DB["password"],
        port=DB["port"],
        options="-c client_encoding=UTF8 -c lc_messages=C",
    )
    return conn

@st.cache_data
def load_all_articles():
    conn = get_psql_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT
            id,
            title,
            summary,
            publish_date,
            category,
            event_loc,
            url
        FROM spnews_summary
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    df = pd.DataFrame(rows)

    return df

# ========================= #
# Streamlit 세팅
# ========================= #
st.set_page_config(page_title="News Data Dashboard", layout="wide")
st.title("NVISIA: North-Korea Vision & Insights by SIA")

if "geo" not in st.session_state:
    st.session_state["geo"] = Geocoder(**DB)
if "rec" not in st.session_state:
    st.session_state["rec"] = Recommender(**DB)

geo: Geocoder = st.session_state["geo"]
rec: Recommender = st.session_state["rec"]

# 클릭된 기사 id / loc 저장
if "selected_id" not in st.session_state:
    st.session_state["selected_id"] = None
    
if "selected_loc" not in st.session_state:
    st.session_state["selected_loc"] = None

# 추천결과 캐싱
if "rec_df" not in st.session_state:
    st.session_state["rec_df"] = None

# ========================= #
# 기타 세팅
# ========================= #
# Matplotlib 한글 폰트 설정 (Windows)
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# =========================
# 데이터 로드
# =========================
df = load_all_articles()

# publish_date 기준 내림차순 정렬 (최신 날짜가 먼저)
if not df.empty and 'publish_date' in df.columns:
    df = df.sort_values('publish_date', ascending=False).reset_index(drop=True)

if df.empty:
    st.warning("No data to display.")
    st.stop()

# =========================
# 레이아웃: 상단
# =========================
top_left, top_right = st.columns([1, 2])

# 동적 콘텐츠를 위한 플레이스홀더
with top_left:
    chart_container = st.empty()

with top_right:
    rec_container = st.empty()

st.divider()


# =========================
# 레이아웃: 하단 
# =========================

if "expanded" not in st.session_state:
    st.session_state.expanded = False

def toggle_expanded():
    st.session_state.expanded = not st.session_state.expanded

st.button(
    "Expand table" if not st.session_state.expanded else "Collapse table",
    on_click=toggle_expanded,
)    

# 높이 결정
table_height = 600 if st.session_state.expanded else 250

# 레이아웃 생성: 왼쪽은 데이터프레임, 오른쪽은 지도
bottom_left, bottom_right = st.columns([2, 1])

# =========================
# 레이아웃: 하단 좌측 테이블
# =========================
with bottom_left:
    # 스크롤 가능한 컨테이너에 전체 데이터프레임 표시
    display_columns = ["id", "title", "summary", "publish_date", "category"]
    # 데이터프레임에 존재하는 컬럼만 포함
    display_columns = [c for c in display_columns if c in df.columns]

    event = st.dataframe(
        df[display_columns],
        height=table_height,
        width="stretch",
        selection_mode="single-row",
        on_select="rerun",
        key="news_table",
    )
    st.caption(f"Showing {len(df)} rows – scroll to view the rest.")

    selected_rows = event["selection"]["rows"]

    # Recommender, Geocoder 에서 모두 활용할 기사 id 를 받는 변수 생성.
    new_id = None
    new_loc = None

    if selected_rows:
        idx = selected_rows[0]
        new_id = df.iloc[idx]["id"]
        new_loc = df.iloc[idx]["event_loc"]

    prev_id = st.session_state["selected_id"]

    # 클릭한 기사가 바뀐 경우에만 다시 계산
    if new_id != prev_id:
        st.session_state["selected_id"] = new_id
        st.session_state["selected_loc"] = new_loc

        if new_id is not None:
            # 추천 새로 계산
            rec_list = rec.get_similar_articles(new_id, k=10)
            rec_df = pd.DataFrame(rec_list)
            if not rec_df.empty:
                rec_df = rec_df.merge(df[["id", "summary"]], on="id", how="left")
            st.session_state["rec_df"] = rec_df
        else:
            st.session_state["rec_df"] = None

selected_id = st.session_state["selected_id"]
selected_loc = st.session_state["selected_loc"]
rec_df = st.session_state["rec_df"]

# =========================
# 레이아웃: 하단 우측 지도
# =========================
with bottom_right:
    if selected_id is not None:
        m = geo.get_map_single(selected_id)
        st_folium(m, width=300, height=400)
    else:
        st.info("위치를 조회하고자 하는 기사를 선택해주세요.")


# ========================= #
# 레이아웃: 상단 우측 추천 뉴스
# ========================= #
if rec_df is not None and not rec_df.empty:
    chart_df = rec_df
    chart_title = "추천 뉴스 카테고리"

    with rec_container.container():
        st.subheader(f"관련 추천 뉴스 (기준: {selected_id})")

        cols = ["id", "title", "summary", "category", "publish_date"]
        cols = [c for c in cols if c in rec_df.columns]

        # 긴 텍스트 축약을 위한 복사본 생성
        display_df = rec_df[cols].copy()
        if "summary" in display_df.columns:
            display_df["summary"] = display_df["summary"].apply(
                lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x
            )

        # 정렬 가능한 컬럼을 위해 st.dataframe 사용
        st.dataframe(
            display_df, 
            width="stretch", 
            hide_index=True, 
            height=300
        )

else:
    # 선택된 행 없음: 전체 데이터를 차트에 사용
    chart_df = df
    chart_title = "전체 뉴스 카테고리"
    rec_container.info("아래 목록에서 기사를 선택하면 추천 뉴스가 표시됩니다.")

# ========================= #
# 레이아웃: 상단 좌측 파이차트
# ========================= #

with chart_container.container():
    if "category" in chart_df.columns:
        st.subheader(chart_title)
        category_counts = chart_df["category"].value_counts()

        if not category_counts.empty:
            def autopct_filter(pct):
                return "%1.1f%%" % pct if pct > 5 else ""

            fig, ax = plt.subplots(figsize=(1.7, 1.7))
            wedges, texts, autotexts = ax.pie(
                category_counts,
                labels=category_counts.index,
                autopct=autopct_filter,
                startangle=90,
                textprops={"fontsize": 4},
            )
            for autotext in autotexts:
                autotext.set_fontsize(4)
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("No category data to display.")