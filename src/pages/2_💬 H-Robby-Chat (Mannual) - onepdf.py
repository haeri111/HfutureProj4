import os
import streamlit as st
from io import StringIO
import re
import sys
from modules.history import ChatHistory
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar
import pdfplumber
import pypdf
import tempfile
import io
from dotenv import load_dotenv

# OpenAI의 언어 모델을 사용하여 단어 임베딩을 생성
from langchain_community.embeddings import OpenAIEmbeddings

# Python에서 PDF 파일을 로드하는 클래스
from langchain.document_loaders import PyPDFLoader

# 디렉토리에서 여러 개의 PDF 파일을 로드하는 클래스
from langchain.document_loaders import PyPDFDirectoryLoader

# 텍스트를 문자 단위로 분할
from langchain.text_splitter import CharacterTextSplitter

# 벡터 데이터를 저장하고 쿼리할 수 있는 인터페이스를 제공
from langchain_chroma import Chroma

# 대화 템플릿 생성
from langchain_core.prompts import ChatPromptTemplate

# 문자열 출력을 파싱하는 클래스
from langchain.schema.output_parser import StrOutputParser

# 함수를 wrapping >> chaining 가능하게 하는 클래스
from langchain.schema.runnable import RunnablePassthrough
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

ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Utilities = utils_module.Utilities
Sidebar = sidebar_module.Sidebar

st.set_page_config(layout="wide", page_icon="💬", page_title="H-Robby | 1pdf-Chat-Bot 🤖")

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


# .env 파일 활성화
load_dotenv()
# user_api_key = utils.load_api_key()
user_api_key = os.getenv('OPENAI_API_KEY')

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chat_with_user(user_message):
    ai_message = chain.invoke(user_message)
    return ai_message

if not user_api_key:
    layout.show_api_key_missing()
else:
    os.environ["OPENAI_API_KEY"] = user_api_key

    # uploaded_file = utils.handle_upload(["pdf", "txt", "csv"])

    # if uploaded_file:

    # folder_path = '/content/drive/MyDrive/2024 KOSA 프로젝트 4조/지식등록'
    # folder_path = '/content/drive/MyDrive/2024 KOSA 프로젝트 4조/지식등록'
    folder_path = '/Users/haeri/Downloads/지식등록/'
    texts = []

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=50
    )

    # 문서 읽기 >> 청크로 나누기 >> 임베딩 >> DB 저장

    filename = '★(ITSM-P-002) 장애 관리 프로세스6.0_00.pdf'
    # for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        raw_documents = PyPDFLoader(folder_path + '/' + filename).load()

    # 텍스트 청크로 나누기
    documents = text_splitter.split_documents(raw_documents)
    texts.extend(documents)
    # st.write(raw_documents)

    utils.show_pdf_file2(raw_documents)

    # texts에는 분할된 청크들이 저장
    st.write(f"처리된 텍스트 청크: {len(texts)}개")

    print("#"*30)
    print(len(texts))
    print(texts)
    print("#"*30)

    # Configure the sidebar
    sidebar.show_options()
    sidebar.about()

    # Initialize chat history
    history = ChatHistory()

    # 로컬 디렉토리 경로 설정
    persist_directory = "./chroma_db"  # 디렉토리 경로 지정

    db = Chroma.from_documents(texts, OpenAIEmbeddings(), persist_directory=persist_directory)
    retriever = db.as_retriever(search_kwargs={"k": 10})

    template = """
    Your role is to use the designated data source, which only includes information related to Hyundai Futurenet or previously Hyundai IT&E, to generate accurate and detailed answers to user inquiries. All information should be explained in Korean based on the retrieved search results and must be factual. Each answer must include all sources used to generate the response, and whenever possible, provide clickable links. There should be no limit to the number of sources, and all relevant sources should be included.

    ### Specific Guidelines:
    - **Friendly Greetings**: Greet the user only at the start of the conversation. Begin the first response with a warm and friendly greeting, such as "안녕하세요! 질문해 주셔서 감사합니다." or "안녕하세요! 도움을 드리게 되어 기쁩니다." After the initial greeting, avoid repeating greetings in subsequent responses. You should indicate that you are only giving information about "현대 퓨처넷"
    - **Exclude unrelated information**: Strictly adhere to information found in the designated data source, and exclude any information that is not directly supported by the search results.
    - **Handling unrelated search results**: If the search results are not related to the user's query, clearly state that the information could not be found and use the following fallback response: "죄송합니다. 요청하신 정보를 찾을 수 없습니다."
    - **Review the user's query**: Thoroughly review the user's query to ensure it is directly related to the search results. If the query is unrelated, use the fallback response.

    ### Formatting Guidelines:
    - Use **bold** for section titles.
    - Apply indentation rules for better readability.
    - Include sources as clickable links at the end of your response, where possible. Include all the sources used and do not limit the number of sources. Include both document titles and URLs as sources where applicable.
    - Be careful, as links may contain spaces or special characters that need to be handled properly.
    - **Image inclusion**: If the answer includes relevant keys in the `![description](url)` format, include corresponding image links at appropriate locations within the answer. This is of utmost importance. Place the images using the `![description](url)` format with descriptions.

    ### Language and Tone:
    - Use a friendly and professional tone.
    - Ensure the language is clear and concise.
    - Adjust the level of detail based on the complexity of the question.

    ### Answer Length:
    - Provide long and detailed answers for all questions, ensuring thorough explanations and comprehensive coverage of the topic.

    **Additional Notes**:
    - If possible, include relevant images or diagrams to enhance the user's understanding.
    - Encourage users to ask follow-up questions if they need further clarification.
    - If the answer includes relevant keys in the `![description](url)` format, include the corresponding image links in the appropriate locations within the answer.

    # Critical: Citation and Source Handling Guidelines
    1. Providing sources:
    - Do not search information from the internet; search only from the provided documents.
    - Always provide sources for the information used in your answers.
    - Source information should be included in the 'source' or 'location' field.

    2. Link format:
    - If the source includes a URL, always provide it as a clickable markdown link.
    - Link format: `[Title](URL)`
    - Example: [OpenAI](https://www.openai.com)

    3. Title processing:
    - If the title contains square brackets ([]), remove them.
    - Example: "GPT-4 [Latest Version]" → "GPT-4 Latest Version"

    4. Non-URL sources:
    - If the source information is not in URL format, display it as is.
    - Example: "2023 AI Trends Report, p.45"
    - Sources that are PDF files, only show name of the file not directory path.

    5. Consistency:
    - Provide all source information at the end of the answer in a consistent format.
    - Preface the source information with the word "**Sources:**" for clarity.

    Answer in Korean.

    [내용] -- 시작 --

    {context}

    [내용] -- 끝 --

    [질문]
    {question}
    """

    try:
        # Initialize chat history
        history = ChatHistory()

        # 정의된 템플릿 가지고 통신하게 해줘야지
        chat_prompt_template = ChatPromptTemplate.from_template(template)
        # chat_model = ChatOpenAI(model_name="gpt-4o-mini")
        chat_model = ChatOpenAI(model = st.session_state["model"],
                                temperature = st.session_state["temperature"])

        st.session_state["ready"] = True
        # chatbot = utils.setup_chatbot(
        #     uploaded_file, st.session_state["model"], st.session_state["temperature"]
        # )

        # st.session_state["chatbot"] = chatbot
        st.session_state["chatbot"] = chat_model

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | chat_prompt_template
            | chat_model
            | StrOutputParser()
        )

        if st.session_state["ready"]:
            # Create containers for chat responses and user prompts
            response_container, prompt_container = st.container(), st.container()

            with prompt_container:
                # Display the prompt form
                is_ready, user_input = layout.prompt_form()

                # Initialize the chat history
                # history.initialize(uploaded_file)
                history.initialize2(filename)

                # Reset the chat history if button clicked
                if st.session_state["reset_chat"]:
                    history.reset2(filename)
                    # history.reset(uploaded_file)

                if is_ready:
                    # Update the chat history and display the chat messages
                    history.append("user", user_input)

                    old_stdout = sys.stdout
                    sys.stdout = captured_output = StringIO()

                    # output = st.session_state["chatbot"].conversational_chat(user_input)
                    output = chat_with_user(user_input)

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
