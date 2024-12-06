import streamlit as st

# Config
st.set_page_config(layout="wide", page_icon="💬", page_title="H-Robby | 매뉴얼 Chat-Bot 🤖")

# Custom CSS for font style
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans&family=Black+Han+Sans&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Do+Hyeon&family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Do+Hyeon&family=IBM+Plex+Sans+KR&family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');
        
        /* Global Style */
        body {
            font-family: 'Open Sans', sans-serif;
        }

        h2 {
            font-family: 'IBM Plex Sans KR', sans-serif;
        }

        h1, h3, h4, h5 {
            font-family: 'IBM Plex Sans KR', sans-serif;
        }

        /* Sidebar */
        .sidebar .sidebar-content {
            font-family: 'Open Sans', sans-serif;
        }

        /* Heading Styling */
        h2 {
            font-size: 36px;
            color: #4CAF50;
        }

        h5 {
            font-size: 18px;
            color: #333;
        }

        p {
            font-size: 18px;
            color: #333;
            font-family: 'IBM Plex Sans KR', sans-serif;
        }

        /* Subheader */
        .css-1f3o0kk {
            font-family: 'Do Hyeon', sans-serif;
            font-size: 20px;
        }

        .css-1f3o0kk > div > div > div {
            line-height: 2;
        }

        /* List styling */
        ul {
            line-height: 1.8;
        }

        li {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)



# Contact
with st.sidebar.expander("📬 Developer"):
    st.write("**한성진:** 매뉴얼업로드UI 개발 및 매뉴얼청킹")
    st.write("**이가영:** 매뉴얼임베딩 및 벡터DB연동")
    st.write("**신하랑:** RAG구성 (검색 및 질문응답 연동)")
    st.write("**최상연:** 프롬프트 템플릿 최적화 및 튜닝")
    st.write("**성해리:** 채팅UI 개발 및 챗봇 문맥 유지")

# Title
st.markdown("""
    <h2 style='text-align: center;'>H-Robby, 현대퓨처넷 매뉴얼 관리 비서! 🤖</h2>
""", unsafe_allow_html=True)

st.markdown("---")

# Description
st.markdown("""
    <p style='text-align:center;'>안녕하세요! 저는 H-Robby입니다. <br>
    현대퓨처넷의 사내 매뉴얼을 관리하고, 필요한 정보를 빠르게 찾아주는 똑똑한 챗봇이에요. <br>
    LangChain과 Streamlit의 강점을 활용해 매뉴얼을 업로드하고, 질문을 통해 매뉴얼을 검색하거나 챗봇과 대화할 수 있습니다. 😎</h5>
""", unsafe_allow_html=True)

st.markdown("---")

# Robby's Pages
st.subheader("🚀 H-Robby가 제공하는 주요 기능")
st.write("""
- **매뉴얼 업로드 📂**: UI를 통해 매뉴얼 파일을 간단히 업로드하세요! 
    - 지원 포맷: PDF, DOCX 등 다양한 파일 형식
    - 스마트 검증: 파일 형식과 크기를 확인해 업로드 가능 여부를 알려드려요.
    - 데이터 저장: 업로드된 매뉴얼은 자동으로 데이터베이스에 저장됩니다.
- **매뉴얼 청킹 및 벡터화 🧠**: 업로드된 매뉴얼을 분석해 빠르고 효율적인 검색이 가능하도록 준비합니다.
    - 매뉴얼을 작은 청크 단위로 분할
    - 청크를 임베딩 모델로 벡터화 후 벡터DB에 저장
    - 벡터 기반 유사도 검색으로 사용자 질문에 정확히 대응
- **벡터 기반 검색 🔍**: 원하는 정보를 키워드만으로 간단히 찾아보세요!
    - 입력한 검색어와 관련된 매뉴얼 청크를 벡터 유사도로 빠르게 탐색
    - 검색 결과는 관련성 높은 순으로 정렬
    - 요약된 내용과 함께 원본 문서 링크 제공
- **질문 응답 챗봇 💬**: 챗봇에게 매뉴얼 관련 질문을 하면 적절한 답변을 제공합니다.
    - 질문 내용을 벡터화해 매뉴얼과 연결
    - 요약된 답변과 관련 매뉴얼 링크를 제공
    - 문맥 유지 기능으로 대화의 흐름을 이어갑니다.
- **매뉴얼 업데이트 및 동기화 🔄**: 최신 정보가 반영될 수 있도록 자동 업데이트를 지원합니다.
    - 새로운 매뉴얼 업로드 시 기존 데이터와 동기화
    - 변경된 내용은 즉시 데이터베이스에 반영
    - 항상 최신 정보로 검색 및 질문 응답 가능
""")
st.markdown("---")

# Contributing
st.markdown("### 🎯 H-Robby와 함께하세요!")
st.markdown("""
매뉴얼 관리가 번거로우셨나요? 반복되는 질문에 지치셨나요? <br>
Robby가 더 똑똑하고 효율적인 방식으로 해결해 드립니다. <br>
함께하면 업무가 훨씬 편리해집니다! 😊
""", unsafe_allow_html=True)