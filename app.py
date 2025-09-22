import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import joblib
import sys

# 모델 관련 모듈 임포트
try:
    from model import BidRatePredictor
    from data_processor import DataProcessor
    from train_model import train_and_save_model
except ImportError:
    st.error("필요한 모듈을 찾을 수 없습니다. 파일을 확인해주세요.")
    sys.exit(1)

# 페이지 설정 - 반응형 웹 최적화
st.set_page_config(
    page_title="사정율 예측 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# 사정율 예측 시스템 v1.0"
    }
)

# 모바일 반응형 CSS
st.markdown("""
<style>
    /* 모바일 최적화 CSS */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.5rem !important;
        }

        .stButton > button {
            width: 100%;
            height: 3rem;
            font-size: 1.1rem !important;
        }

        .metric-container {
            padding: 0.5rem !important;
        }

        h1 {
            font-size: 1.5rem !important;
        }

        h2 {
            font-size: 1.2rem !important;
        }

        h3 {
            font-size: 1rem !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem !important;
            font-size: 0.9rem !important;
        }

        /* 입력 필드 모바일 최적화 */
        .stNumberInput > div {
            width: 100% !important;
        }

        .stSelectbox > div {
            width: 100% !important;
        }

        /* 차트 모바일 최적화 */
        .plotly-graph-div {
            width: 100% !important;
        }

        /* 사이드바 모바일 최적화 */
        section[data-testid="stSidebar"] {
            width: 80% !important;
        }

        /* 메트릭 카드 모바일 최적화 */
        [data-testid="metric-container"] {
            padding: 0.5rem !important;
            margin: 0.25rem !important;
        }
    }

    /* 태블릿 최적화 */
    @media (min-width: 768px) and (max-width: 1024px) {
        .block-container {
            padding: 1rem !important;
        }

        .stButton > button {
            width: auto;
            min-width: 150px;
        }
    }

    /* 공통 스타일 */
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .prediction-result {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }

    .info-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    /* 스크롤 개선 */
    .main {
        overflow-x: hidden;
    }

    /* 터치 친화적 버튼 */
    button {
        min-height: 44px;
        min-width: 44px;
    }

    /* 네비게이션 바 */
    .navbar {
        position: sticky;
        top: 0;
        z-index: 999;
        background: white;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# 헤더
st.markdown("""
<div class="main-header">
    <h1>📊 사정율 예측 시스템</h1>
    <p>입찰 데이터 분석 및 예측 플랫폼</p>
</div>
""", unsafe_allow_html=True)

# 모바일 친화적 탭 메뉴
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 홈", "📊 분석", "🎯 예측", "📈 시각화", "⚙️ 설정"])

# 홈 탭
with tab1:
    # 반응형 컬럼 레이아웃
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📌 주요 기능</h3>
            <ul>
                <li>실시간 사정율 예측</li>
                <li>데이터 시각화 분석</li>
                <li>통계 리포트 생성</li>
                <li>배치 예측 처리</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📊 데이터 현황</h3>
        </div>
        """, unsafe_allow_html=True)

        # 데이터 로드 상태 확인
        if os.path.exists('강원도및경기일부.xlsx'):
            if st.button("📂 데이터 로드", use_container_width=True):
                with st.spinner('데이터 로딩 중...'):
                    try:
                        st.session_state.data = pd.read_excel('강원도및경기일부.xlsx')
                        st.success("✅ 데이터 로드 완료!")
                    except Exception as e:
                        st.error(f"❌ 데이터 로드 실패: {e}")

    # 데이터 요약 정보
    if st.session_state.data is not None:
        st.markdown("### 📊 데이터 요약")

        # 반응형 메트릭 표시
        metrics = st.columns([1, 1, 1])
        with metrics[0]:
            st.metric("전체 데이터", f"{len(st.session_state.data):,}건")
        with metrics[1]:
            st.metric("컬럼 수", f"{len(st.session_state.data.columns)}개")
        with metrics[2]:
            valid_rate = (1 - st.session_state.data.isnull().sum().sum() /
                         (len(st.session_state.data) * len(st.session_state.data.columns))) * 100
            st.metric("데이터 완성도", f"{valid_rate:.1f}%")

# 분석 탭
with tab2:
    st.markdown("### 📊 데이터 분석")

    if st.session_state.data is not None:
        # 데이터 미리보기
        with st.expander("📋 데이터 미리보기", expanded=False):
            st.dataframe(
                st.session_state.data.head(10),
                use_container_width=True,
                height=300
            )

        # 통계 요약
        with st.expander("📈 기본 통계", expanded=True):
            st.write(st.session_state.data.describe())

        # 결측치 분석
        with st.expander("🔍 데이터 품질 분석", expanded=False):
            missing = st.session_state.data.isnull().sum()
            missing_df = pd.DataFrame({
                '컬럼': missing.index,
                '결측치': missing.values,
                '비율(%)': (missing.values / len(st.session_state.data)) * 100
            })
            missing_df = missing_df[missing_df['결측치'] > 0]

            if not missing_df.empty:
                fig = px.bar(missing_df, x='컬럼', y='비율(%)',
                           title="결측치 현황",
                           color='비율(%)',
                           color_continuous_scale='Reds')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ 결측치가 없습니다!")
    else:
        st.info("📂 먼저 홈 탭에서 데이터를 로드해주세요.")

# 예측 탭
with tab3:
    st.markdown("### 🎯 사정율 예측")

    # 입력 폼 - 모바일 최적화
    with st.form("prediction_form"):
        st.markdown("#### 입력 데이터")

        # 반응형 입력 레이아웃
        col1, col2 = st.columns([1, 1])

        with col1:
            기초금액 = st.number_input(
                "기초금액 (원)",
                min_value=0,
                value=100000000,
                step=1000000,
                format="%d"
            )

            예정가격 = st.number_input(
                "예정가격 (원)",
                min_value=0,
                value=95000000,
                step=1000000,
                format="%d"
            )

        with col2:
            투찰업체수 = st.number_input(
                "투찰업체수",
                min_value=1,
                value=10,
                step=1
            )

            지역 = st.selectbox(
                "지역",
                ["강원도", "경기도", "서울", "기타"]
            )

        # 예측 버튼
        submitted = st.form_submit_button(
            "🎯 예측하기",
            use_container_width=True,
            type="primary"
        )

        if submitted:
            with st.spinner('예측 중...'):
                # 여기서 실제 모델 예측 수행
                # 임시로 랜덤 값 사용
                예측값 = np.random.uniform(85, 95)
                하한 = 예측값 - np.random.uniform(1, 3)
                상한 = 예측값 + np.random.uniform(1, 3)

                # 예측 결과 표시
                st.markdown("""
                <div class="prediction-result">
                    <h3>예측 결과</h3>
                    <div class="prediction-value">{:.2f}%</div>
                    <p>신뢰구간: {:.2f}% ~ {:.2f}%</p>
                </div>
                """.format(예측값, 하한, 상한), unsafe_allow_html=True)

                # 예측 이력 저장
                st.session_state.predictions.append({
                    '시간': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    '예측값': 예측값,
                    '기초금액': 기초금액,
                    '투찰업체수': 투찰업체수
                })

# 시각화 탭
with tab4:
    st.markdown("### 📈 데이터 시각화")

    if st.session_state.data is not None:
        # 차트 선택
        chart_type = st.selectbox(
            "차트 유형 선택",
            ["히스토그램", "산점도", "박스플롯", "히트맵"]
        )

        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()

        if chart_type == "히스토그램":
            if numeric_cols:
                selected_col = st.selectbox("컬럼 선택", numeric_cols)
                fig = px.histogram(
                    st.session_state.data,
                    x=selected_col,
                    title=f"{selected_col} 분포",
                    nbins=30
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "산점도":
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X축", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y축", numeric_cols)

                fig = px.scatter(
                    st.session_state.data,
                    x=x_col,
                    y=y_col,
                    title=f"{x_col} vs {y_col}"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📂 먼저 홈 탭에서 데이터를 로드해주세요.")

# 설정 탭
with tab5:
    st.markdown("### ⚙️ 설정")

    # 테마 설정
    st.markdown("#### 🎨 테마 설정")
    theme = st.radio(
        "테마 선택",
        ["라이트", "다크", "자동"],
        horizontal=True
    )

    # 알림 설정
    st.markdown("#### 🔔 알림 설정")
    notification = st.checkbox("예측 완료 알림", value=True)

    # 데이터 설정
    st.markdown("#### 💾 데이터 설정")
    if st.button("🗑️ 캐시 초기화", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ 캐시가 초기화되었습니다.")

    # 버전 정보
    st.markdown("#### ℹ️ 버전 정보")
    st.info("""
    **사정율 예측 시스템**
    - 버전: 1.0.0
    - 최종 업데이트: 2024.01
    - 개발: BidRate Predictor Team
    """)

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    © 2024 BidRate Predictor. All rights reserved.
</div>
""", unsafe_allow_html=True)