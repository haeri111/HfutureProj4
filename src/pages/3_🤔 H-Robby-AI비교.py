import streamlit as st
import os
import urllib.parse
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from transformers import pipeline, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from modules.chatbot2 import Chatbot2
from langchain_community.embeddings import OpenAIEmbeddings # OpenAI의 언어 모델을 사용하여 단어 임베딩을 생성
from langchain_chroma import Chroma # 벡터 데이터를 저장하고 쿼리할 수 있는 인터페이스를 제공
# import llm_blender
# from transformers import AutoTokenizer, AutoModelForCausalLM
from modules.history import ChatHistory
from modules.chromaManager import ChromaManager

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# OpenAI API 키 로드
openai.api_key = os.getenv('OPENAI_API_KEY')

# 페이지 설정
st.set_page_config(layout="wide", page_icon="🤖", page_title="AI 모델 비교")

history = ChatHistory()

# 질의 입력 UI
st.markdown(
    """
    <head>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&display=swap" rel="stylesheet">
    </head>
    <h1 style='text-align: center; font-family: "IBM Plex Sans KR", sans-serif;'> H-Robby가 AI 모델을 비교해드려요 🤔</h1>
    """,
    unsafe_allow_html=True,
)

st.write("질문을 입력하고 각 모델의 응답 및 비용을 비교해보세요.")
user_query = st.text_input("질문 입력", placeholder="예: 현대퓨처넷은 어디에 위치해있나요?")

# Chatbot2 객체 초기화
# retriever = None  # Retriever를 필요에 따라 초기화
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

persist_directory = "./chroma_db"  # 로컬 디렉토리 경로 지정
manager = ChromaManager(persist_directory=persist_directory)    # client, retriever 초기화 및 db 로드
retriever = manager.get_retriever()

openai_chatbot = Chatbot2("OpenAI", "gpt-4", 0.7, retriever)
gemini_chatbot = Chatbot2("Gemini", "gemini-pro", 0.7, retriever)
huggingface_chatbot = Chatbot2("HuggingFace", "gpt2", 0.7, retriever)

# 모델 응답 컨테이너
if user_query:
    with st.spinner("모델 응답을 생성 중..."):
        # OpenAI 응답
        openai_response = openai_chatbot.chat(user_query)
        openai_cost = len(user_query) * 0.02  # 예시 비용 계산

        # Gemini 응답
        gemini_response = gemini_chatbot.chat(user_query)
        gemini_cost = len(user_query) * 0.015  # 예시 비용 계산

        # HuggingFace 응답
        huggingface_response = huggingface_chatbot.chat(user_query)
        # huggingface_cost = len(user_query) * 0.01  # 예시 비용 계산


        # 결과 표시
        st.markdown("## 모델 응답 비교")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### OpenAI")
            st.text_area("OpenAI응답", value=openai_response["answer"], height=350)
            # st.write(f"비용: ${333:.4f}")
            st.write(f"비용: ${openai_cost:.4f}")
            # Display metadata in expander
            # 중복 제거
            retrieved_documents = openai_response["documents"]
            unique_documents = list({(doc.metadata.get("source"), doc.metadata.get("page")): doc for doc in retrieved_documents}.values())
            with st.expander("관련 매뉴얼 링크 🔗"):
                # retrieve_context에서 사용된 context 데이터를 저장
                if unique_documents:
                    st.subheader("관련 매뉴얼")
                    for doc in unique_documents:
                        metadata = doc.metadata
                        source = metadata.get("source", "Unknown Source")
                        page = metadata.get("page", "Unknown Page")

                        # 클릭 가능한 링크로 변환
                        format_source = f"file://{urllib.parse.quote(source)}"  # 경로 변환

                        st.markdown(f"- [Source: {source} (Page {page})]({format_source})", unsafe_allow_html=True)
                else:
                    st.write("메타데이터 정보가 없습니다.")

        with col2:
            st.markdown("### Gemini")
            st.text_area("Gemini응답", value=gemini_response["answer"], height=350)
            # st.write(f"비용: ${222:.4f}")
            st.write(f"비용: ${gemini_cost:.4f}")
            # Display metadata in expander
            # 중복 제거
            retrieved_documents = gemini_response["documents"]
            unique_documents = list({(doc.metadata.get("source"), doc.metadata.get("page")): doc for doc in retrieved_documents}.values())
            with st.expander("관련 매뉴얼 링크 🔗"):
                # retrieve_context에서 사용된 context 데이터를 저장
                if unique_documents:
                    st.subheader("관련 매뉴얼")
                    for doc in unique_documents:
                        metadata = doc.metadata
                        source = metadata.get("source", "Unknown Source")
                        page = metadata.get("page", "Unknown Page")

                        # 클릭 가능한 링크로 변환
                        format_source = f"file://{urllib.parse.quote(source)}"  # 경로 변환

                        st.markdown(f"- [Source: {source} (Page {page})]({format_source})", unsafe_allow_html=True)
                else:
                    st.write("메타데이터 정보가 없습니다.")

        with col3:
            st.markdown("### HuggingFace")
            st.text_area("HuggingFace응답", value=huggingface_response, height=350)
            st.write(f"비용: free")
            # st.write(f"비용: ${huggingface_cost:.4f}")



# 앙상블 설정 가이드
# st.sidebar.markdown(
#     """
#     ### 📌 가이드
#     - 각 모델의 응답 가중치를 조정해 앙상블 응답 결과를 변경할 수 있습니다.
#     - 가중치 합은 **1.0**이 되어야 가장 정확한 결과를 기대할 수 있습니다.
#     """
# )
