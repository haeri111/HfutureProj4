import os
import streamlit as st
from io import StringIO
import re
import sys
from modules.history import ChatHistory
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar
from modules.googledrive import GoogleDrive
from modules.chromaManager import ChromaManager
from modules.prompt import PROMPT_STR
from modules.chatbot2 import Chatbot2
import pdfplumber
import pypdf
import tempfile
import io
from dotenv import load_dotenv
# from chromadb import Client
# from chromadb.config import Settings
from chromadb import PersistentClient
from langchain_community.embeddings import OpenAIEmbeddings # OpenAI의 언어 모델을 사용하여 단어 임베딩을 생성
from langchain.text_splitter import CharacterTextSplitter # 텍스트를 문자 단위로 분할
from langchain_chroma import Chroma # 벡터 데이터를 저장하고 쿼리할 수 있는 인터페이스를 제공
from langchain_core.prompts import ChatPromptTemplate # 대화 템플릿 생성
from langchain.schema.output_parser import StrOutputParser # 문자열 출력을 파싱하는 클래스
from langchain.schema.runnable import RunnablePassthrough # 함수를 wrapping >> chaining 가능하게 하는 클래스
from langchain_community.chat_models import ChatOpenAI


# To be able to update the changes made to modules in localhost (press r)
def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]

history_module = reload_module('modules.history')
layout_module = reload_module('modules.layout')
utils_module = reload_module('modules.utils')
sidebar_module = reload_module('modules.sidebar')
googledrive_module = reload_module('modules.googledrive')

ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Utilities = utils_module.Utilities
Sidebar = sidebar_module.Sidebar
GoogleDrive = googledrive_module.GoogleDrive

st.set_page_config(layout="wide", page_icon="💬", page_title="H-Robby | Chat-Bot 🤖")

# 메인컴포넌트 초기화
layout, sidebar, utils = Layout(), Sidebar(), Utilities()

st.markdown(
    f"""
    <head>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&display=swap" rel="stylesheet">
    </head>
    <h1 style='text-align: center; font-family: "IBM Plex Sans KR", sans-serif;'> H-Robby에게 무엇이든 물어보세요! 😁</h1>
    """,
    unsafe_allow_html=True,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chat_with_user(user_message):
    ai_message = chain.invoke(user_message)
    return ai_message

# .env 파일 활성화
load_dotenv()
user_api_key = os.getenv('OPENAI_API_KEY')
gen_api_key = os.getenv('GOOGLE_API_KEY')

if not user_api_key:
    layout.show_api_key_missing()
else:
    os.environ["OPENAI_API_KEY"] = user_api_key
    os.environ["GOOGLE_API_KEY"] = gen_api_key

    # 사이드바 추가
    sidebar.show_options()
    sidebar.about()

    # chat history 객체 초기화
    history = ChatHistory()

    #############################################################################################################
    # Google Drive API 로 공유폴더에 업로드 되어있는 chroma DB 연동하기
    # 1. 서비스 기동 시  chroma.sqlite3 다운받은 경로 기준으로 클라이언트 초기화
    # 2. 이후 테이블 접근 가능
    #############################################################################################################

    # Google Drive 객체 초기화
    service_account_file = '/Users/haeri/Downloads/h-manualone-9e64b070661d.json'
    scopes = ['https://www.googleapis.com/auth/drive']
    googledrive = GoogleDrive(service_account_file, scopes)
    # 폴더 ID 설정
    folder_id = "1sU-NhDxUm4U5c3IVPmfiHmO_1pY4bjLg"

    print("\n\n\n왜 두번 호출되냐고 \n\n\n")

    # 구글 드라이브에서 chroma.sqlite3 파일을 찾고 다운로드
    googledrive.download_chroma_db(folder_id)

    # 초기화
    persist_directory = "./chroma_db2"  # 로컬 디렉토리 경로 지정
    manager = ChromaManager(persist_directory=persist_directory)    # client, retriever 초기화 및 db 로드

    # 컬렉션 초기화
    collection_names = ["user_permissions", "search_history", "file_table", "chunk_table"]
    manager.initialize_collections(collection_names)

    # retriever get
    retriever = manager.get_retriever()
    print(f"********************Retriever : {retriever}")

    manager.debug_collections()
    manager.query_collection("chunk_table","장애 발생 보고 알려줘")

    print("***********************")
    print(retriever.invoke("장애 발생 보고 알려줘"))
    print("***********************")

    template = PROMPT_STR

try:
    # chat history 초기화
    history = ChatHistory()

    # 정의된 템플릿 가지고 통신하게 해줘야지
    chat_prompt_template = ChatPromptTemplate.from_template(template)
    # chat_model = ChatOpenAI(model_name="gpt-4o-mini")
    # chat_model = ChatOpenAI(model = st.session_state["model"],
    #                         temperature = st.session_state["temperature"])

    # 모델 관련 설정
    model_type = st.session_state.get("model_type", "OpenAI")  # 기본값은 openai
    model_name = st.session_state["model"]
    temperature = st.session_state["temperature"]

    # Chatbot 설정
    st.session_state["chatbot"] = Chatbot2(
        model_type, model_name, temperature, retriever
    )
    chatbot = st.session_state.get("chatbot")
    if not chatbot:
        st.error("Chatbot setup failed.")
        st.stop()

    st.session_state["ready"] = True

    # chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | chat_prompt_template
    #     | chat_model
    #     | StrOutputParser()
    # )

    if st.session_state["ready"]:
        # 채팅, 사용자 프롬프트를 위한 컨테이너 생성
        response_container, prompt_container = st.container(), st.container()

        with prompt_container:
            # 프롬프트 폼 초기화
            is_ready, user_input = layout.prompt_form()

            # chat history 초기화
            history.initialize2("현대퓨처넷")

            # 버튼 클릭시 chat history 리셋
            if st.session_state["reset_chat"]:
                history.reset2("현대퓨처넷")

            if is_ready:
                # chat history 업데이트 및 chat messages 노출
                history.append("user", user_input)

                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                # context = retriever.retrieve(user_input)
                # output = st.session_state["chatbot"].conversational_chat(user_input)
                # output = chat_with_user(user_input)
                output = chatbot.chat(user_input)

                sys.stdout = old_stdout

                history.append("assistant", output)

                # Clean up the agent's thoughts to remove unwanted characters
                thoughts = captured_output.getvalue()
                cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)

                # Display the agent's thoughts
                with st.expander("Display the agent's thoughts"):
                    st.write(cleaned_thoughts)

        history.generate_messages(response_container)
except Exception as e:
    st.error(f"Error: {str(e)}")
