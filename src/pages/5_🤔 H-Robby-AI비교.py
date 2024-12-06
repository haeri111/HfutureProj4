import streamlit as st
import os
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from modules.chatbot2 import Chatbot2
# import llm_blender
# from transformers import AutoTokenizer, AutoModelForCausalLM

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# OpenAI API 키 로드
openai.api_key = os.getenv('OPENAI_API_KEY')

# 페이지 설정
st.set_page_config(layout="wide", page_icon="🤖", page_title="AI 모델 비교")

# Chatbot 객체 생성
# chatbot = Chatbot()

# 사용자 정의 가중치 슬라이더 (앙상블용)
# st.sidebar.title("LLM 앙상블 설정")
# openai_weight = st.sidebar.slider("OpenAI 가중치", 0.0, 1.0, 0.33)
# gemini_weight = st.sidebar.slider("Gemini 가중치", 0.0, 1.0, 0.33)
# huggingface_weight = st.sidebar.slider("HuggingFace 가중치", 0.0, 1.0, 0.34)


# # 모델 초기화
# openai_model = ChatOpenAI(model="gpt-4", temperature=0.7)
# gemini_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # Gemini 모델 예시
# # huggingface_model = pipeline(model="bigscience/bloom", temperature=0.7)
# config = GPT2Config.from_pretrained("skt/kogpt2-base-v2")
# huggingface_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# huggingface_model = GPT2LMHeadModel.from_pretrained("gpt2")

# # 템플릿 설정
# chat_prompt_template = ChatPromptTemplate.from_template(prompt_template)

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
retriever = None  # Retriever를 필요에 따라 초기화
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
        huggingface_cost = len(user_query) * 0.01  # 예시 비용 계산


        # # HuggingFace (GPT-2) 응답
        # inputs = huggingface_tokenizer(user_query, return_tensors="pt")
        # outputs = huggingface_model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
        # huggingface_response = huggingface_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # huggingface_cost = len(user_query) * 0.01  # 예시 계산

        # Load the Blender ranker and fuser
        # blender = llm_blender.Blender()
        # blender.loadranker("llm-blender/PairRM")
        # blender.loadfuser("llm-blender/gen_fuser_3b")

        # 앙상블 응답 (가중치 기반)
        # ensemble_response = (
        #     f"{openai_weight * openai_response} + "
        #     f"{gemini_weight * gemini_response} + "
        #     f"{huggingface_weight * huggingface_response}"
        # )

        # 결과 표시
        st.markdown("## 모델 응답 비교")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### OpenAI")
            st.text_area("OpenAI응답", value=openai_response, height=350)
            # st.write(f"비용: ${333:.4f}")
            st.write(f"비용: ${openai_cost:.4f}")

        with col2:
            st.markdown("### Gemini")
            st.text_area("Gemini응답", value=gemini_response, height=350)
            # st.write(f"비용: ${222:.4f}")
            st.write(f"비용: ${gemini_cost:.4f}")

        with col3:
            st.markdown("### HuggingFace")
            st.text_area("HuggingFace응답", value="huggingface_response", height=350)
            # st.write(f"비용: ${111:.4f}")
            st.write(f"비용: ${huggingface_cost:.4f}")

        # st.markdown("## 앙상블 응답")
        # responses = {openai_response, gemini_response, huggingface_response}

        # # Rank responses
        # ranked = blender.rank_with_ref(user_query, list(responses.values()), return_scores=True, batch_size=1)
        
        # # Select the top response
        # top_response = max(ranked[0], key=lambda x: x[1])
        # print(f"Top response: {top_response[0]}")

        # # Optionally fuse responses for a refined answer
        # fused_response = blender.fuse(user_query, list(responses.values()), batch_size=1)

        # st.text_area("앙상블 결과", value=fused_response, height=150)
        # # st.text_area("앙상블 결과", value="ensemble_response", height=150)
        # # st.text_area("앙상블 결과", value=ensemble_response, height=150)

def chat_with_user(user_message):
    ai_message = chain.invoke(user_message)
    return ai_message

# 앙상블 설정 가이드
# st.sidebar.markdown(
#     """
#     ### 📌 가이드
#     - 각 모델의 응답 가중치를 조정해 앙상블 응답 결과를 변경할 수 있습니다.
#     - 가중치 합은 **1.0**이 되어야 가장 정확한 결과를 기대할 수 있습니다.
#     """
# )
