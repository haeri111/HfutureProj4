import os
import uuid
import urllib.parse
import streamlit as st
from io import StringIO
from datetime import datetime
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
from langchain_core.prompts import ChatPromptTemplate # 대화 템플릿 생성


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

def format_local_file_link(path):
    """
    로컬 파일 경로를 Mac에서 클릭 가능한 file:// URL로 변환
    """
    abs_path = os.path.abspath(path)  # 절대 경로로 변환
    return f"file://{urllib.parse.quote(abs_path)}"

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

    # 구글 드라이브에서 chroma.sqlite3 파일을 찾고 다운로드
    googledrive.download_chroma_db(folder_id)

    # 초기화
    persist_directory = "./chroma_db2"  # 로컬 디렉토리 경로 지정
    manager2 = ChromaManager(persist_directory=persist_directory)    # client, retriever 초기화 및 db 로드

    # 컬렉션 초기화
    collection_names = ["user_permissions", "search_history", "file_table", "chunk_table", "vector_table"]
    manager2.initialize_collections(collection_names)


    # 초기화
    persist_directory = "./chroma_db3"  # 로컬 디렉토리 경로 지정
    manager = ChromaManager(persist_directory=persist_directory)    # client, retriever 초기화 및 db 로드

    # retriever get
    retriever = manager.get_retriever()
    print(f"********************Retriever : {retriever}")
    # # retriever 사용 시 임베딩 생성 방식 확인
    # print(f"Embedding function used: {retriever.embedding_function}")

    manager2.debug_collections()
    # manager.query_collection("chunk_table","장애 발생 보고 알려줘")
    # print("@@@@@@@@@@@@@@@@@@@@@@")
    # # 임베딩 함수 초기화
    # embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
    # # 쿼리 임베딩 생성
    # query = "장애 발생 보고 알려줘"
    # query_embedding = embedding_function.embed_query(query)  # 쿼리를 벡터로 변환
    # manager.query_collection("vector_table",query_embedding)
    # print("@@@@@@@@@@@@@@@@@@@@@@")

    # print("***********************")
    # query = "장애 발생 보고 알려줘"
    # retriever.invoke(input=query)
    # # retriever.invoke(query_embedding)
    # print("***********************")

    # vector_store = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
    # retriever2 = vector_store.as_retriever(search_kwargs={"k": 10})
    # query = "장애 발생 보고 알려줘"
    # print(f"!!!!!!!!!!!{retriever2}")
    # print(f"!!!!!!!!!!!22222222{retriever2.invoke(input=query)}")

    template = PROMPT_STR

try:
    # chat history 초기화
    history = ChatHistory()

    # 정의된 템플릿 가지고 통신하게 해줘야지
    # chat_prompt_template = ChatPromptTemplate.from_template(template)
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
                result = chatbot.chat(user_input)
                output = result["answer"]
                retrieved_documents = result["documents"]

                sys.stdout = old_stdout

                history.append("assistant", output)

                # Clean up the agent's thoughts to remove unwanted characters
                thoughts = captured_output.getvalue()
                cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)

                # 중복 제거
                unique_documents = list({(doc.metadata.get("source"), doc.metadata.get("page")): doc for doc in retrieved_documents}.values())
                search_history_documents = []

                # Display metadata in expander
                with st.expander("관련 매뉴얼 링크 🔗"):
                    # retrieve_context에서 사용된 context 데이터를 저장
                    if unique_documents:
                        st.subheader("관련 매뉴얼")
                        for doc in unique_documents:
                            metadata = doc.metadata
                            source = metadata.get("source", "Unknown Source")
                            page = metadata.get("page", "Unknown Page")

                            # 클릭 가능한 링크로 변환
                            format_source = format_local_file_link(source)  # 경로 변환

                            st.markdown(f"- [Source: {source} (Page {page})]({format_source})", unsafe_allow_html=True)

                            document_id = str(uuid.uuid4())

                            search_history_documents.append(
                                {
                                    "documents": [user_input],
                                    "metadatas": {
                                            "srch_text": user_input,  # 검색어
                                            "srch_dtm": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 현재 시간
                                            "file_name": source,  # 관련 파일 경로
                                            "page": page
                                        },
                                    "ids": [document_id]
                                }
                            )
                        manager2.insert_into_collection("search_history", search_history_documents)
                        # manager2.insert_into_collection("search_history", search_history_documents)
                        googledrive.upload_to_drive("./chroma_db2/chroma.sqlite3", "chroma.sqlite3", "1sU-NhDxUm4U5c3IVPmfiHmO_1pY4bjLg")
                            
                    else:
                        st.write("메타데이터 정보가 없습니다.")

                # Display the agent's thoughts
                with st.expander("Langchain 프로세스 🧐"):
                    st.write(cleaned_thoughts)

        history.generate_messages(response_container)
except Exception as e:
    st.error(f"Error: {str(e)}")
