# Streamlit Dashboard to view summary_df_final.csv
# Modified to include dynamic recommendation system based on similarity

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

"""

# ========================= #
DB = dict(
    host="localhost",
    database="nvisiaDb",
    user="postgres",
    password="postgres1202",
    port=5432,
)

@st.cache_resource
def get_rec():
    return Recommender(**DB)

@st.cache_resource
def get_geo():
    return Geocoder(**DB)

rec = get_rec()
geo = get_geo()
# ========================= #


# Set Korean font for Matplotlib (Windows)
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

st.set_page_config(page_title="News Data Dashboard", layout="wide")

st.title("NVISIA: North-Korea Vision & Insights by SIA")

@st.cache_data
def load_all_articles():
    conn = psycopg2.connect(**DB)
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
    return pd.DataFrame(rows)


# Load data
df = load_all_articles()

if df.empty:
    st.warning("No data to display.")
else:
    # Create two columns for the top section with 1:2 ratio
    col1, col2 = st.columns([1, 2])

    # Placeholders for dynamic content
    with col1:
        chart_container = st.empty()

    with col2:
        rec_container = st.empty()

    st.divider()

    # Initialize expand state
    if "expanded" not in st.session_state:
        st.session_state.expanded = False

    # Button to toggle expansion
    def toggle_expanded():
        st.session_state.expanded = not st.session_state.expanded

    st.button(
        "Expand table" if not st.session_state.expanded else "Collapse table",
        on_click=toggle_expanded
    )
    
    # Determine height
    table_height = 600 if st.session_state.expanded else 250

    # Show full dataframe in a scrollable container with the calculated height
    # Enable row selection
    # Exclude embedding column from display and specify column order
    display_columns = ['id', 'title', 'summary', 'publish_date', 'category']   

    # Only include columns that exist in the dataframe
    display_columns = [col for col in display_columns if col in df.columns]

    event = st.dataframe(
        df[display_columns], 
        height=table_height, 
        use_container_width=True, 
        on_select="rerun", 
        selection_mode="single-row"
    )

    st.caption(f"Showing {len(df)} rows – scroll to view the rest.")

    # Determine which data to use for the chart and update col2

    if len(event.selection.rows) > 0:
        # Row selected: Get recommendations based on similarity
        selected_idx = event.selection.rows[0]
        selected_id = df.iloc[selected_idx]["id"]

        # rec.py
        rec_list = rec.get_similar_articles(selected_id, k=10)
        rec_df = pd.DataFrame(rec_list)

        if not rec_df.empty:
            rec_df = rec_df.merge(
                df[['id', 'summary']],
                on='id',
                how='left'
            )

        chart_df = rec_df
        chart_title = "추천 뉴스 카테고리"
        
        # Show Recommended Data Table in col2
        with rec_container.container():
            st.subheader(f"관련 추천 뉴스 (기준: {selected_id})")
            if not rec_df.empty:
                # Create a copy for display to truncate long text
                cols_for_display = ['id', 'title', 'summary', 'category', 'publish_date']
                cols_for_display = [c for c in cols_for_display if c in rec_df.columns]
                display_df = rec_df[cols_for_display].copy()

                if 'summary' in display_df.columns:
                    display_df['summary'] = display_df['summary'].apply(lambda x: x[:50] + '...' if isinstance(x, str) and len(x) > 50 else x)
                
                # Use st.dataframe for sortable columns
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=300
                )
            else:
                st.info("No recommended data available.")
    else:
        # No row selected: Use Full Data for Chart
        chart_df = df
        chart_title = "전체 뉴스 카테고리"
        
        # Show info message in col2
        rec_container.info("아래 목록에서 기사를 선택하면 추천 뉴스가 표시됩니다.")

    # Draw Pie Chart in col1
    with chart_container.container():
        if 'category' in chart_df.columns:
            st.subheader(chart_title)
            category_counts = chart_df['category'].value_counts()
            
            if not category_counts.empty:
                def autopct_filter(pct):
                    return ('%1.1f%%' % pct) if pct > 5 else ''
                    
                # Small figsize
                fig, ax = plt.subplots(figsize=(1.7, 1.7)) 
                # Labels outside, no rotation
                wedges, texts, autotexts = ax.pie(
                    category_counts, 
                    labels=category_counts.index, 
                    autopct=autopct_filter, 
                    startangle=90, 
                    textprops={'fontsize': 4}
                )
                
                # Small font size for percentages inside the pie
                for autotext in autotexts:
                    autotext.set_fontsize(4)
                    
                ax.axis('equal')
                st.pyplot(fig, use_container_width=False)
            else:
                st.info("No category data to display.")
