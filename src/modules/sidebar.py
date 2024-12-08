import streamlit as st
import os
import shutil
import modules.embedder as Embedder

class Sidebar:

    LLM_OPTIONS = ["OpenAI", "Gemini", "HuggingFace"]
    MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4o"]
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.0
    TEMPERATURE_STEP = 0.01

    @staticmethod
    def about():
        about = st.sidebar.expander("🧠 About H-Robby ")
        sections = [
            "##### 현대퓨처넷의 사내 매뉴얼을 관리하고, 필요한 정보를 빠르게 찾아주는 똑똑한 챗봇이에요. 📄",
            "##### LangChain과 Streamlit의 강점을 활용해 매뉴얼을 업로드하고, 질문을 통해 매뉴얼을 검색하거나 챗봇과 대화할 수 있습니다. 😎 ",
            "##### [Langchain](https://github.com/hwchase17/langchain), [Streamlit](https://github.com/streamlit/streamlit), [OpenAI](https://platform.openai.com/docs/models/gpt-3-5) 를 사용했어요! ⚡",
        ]
        for section in sections:
            about.write(section)

    @staticmethod
    def reset_chat_button():
        if st.button("Reset chat"):
            st.session_state["reset_chat"] = True
        st.session_state.setdefault("reset_chat", False)

    @staticmethod
    def embed_documents_button(folder_path, persist_directory):
        if st.button("Embed New Documents"):
            embedder = Embedder.Embedder
            embedder.embed_new_documents(folder_path, persist_directory)
            st.session_state["embedding_triggered"] = True
        else:
            st.session_state["embedding_triggered"] = False

    def delete_db(self, persist_directory):
        # 기존 데이터베이스 삭제
        if os.path.exists(persist_directory):
            print(f"기존 데이터베이스({persist_directory})를 삭제 중...")
            shutil.rmtree(persist_directory)  # 디렉토리 및 모든 내용 삭제
            print("삭제 완료!")

    def delete_db_button(self, persist_directory):
        if st.button("Delete DB"):
            self.delete_db(persist_directory)
            st.session_state["embedding_triggered"] = True
        else:
            st.session_state["embedding_triggered"] = False

    def llm_selector(self):
        llm = st.selectbox(label="LLM", options=self.LLM_OPTIONS)
        st.session_state["model_type"] = llm

    def model_selector(self):
        model = st.selectbox(label="Model", options=self.MODEL_OPTIONS)
        st.session_state["model"] = model

    def temperature_slider(self):
        temperature = st.slider(
            label="Temperature",
            min_value=self.TEMPERATURE_MIN_VALUE,
            max_value=self.TEMPERATURE_MAX_VALUE,
            value=self.TEMPERATURE_DEFAULT_VALUE,
            step=self.TEMPERATURE_STEP,
        )
        st.session_state["temperature"] = temperature
        
    def retriever_weight_sliders(self):
        with st.sidebar.expander("⚙️ Retriever 가중치", expanded=False):
            bm25_weight = st.slider(
                label="BM25 Retriever Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
            )
            chroma_weight = st.slider(
                label="Chroma Retriever Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
            )

            # Normalize weights to ensure sum is 1
            total_weight = bm25_weight + chroma_weight
            st.session_state["bm25_weight"] = bm25_weight / total_weight
            st.session_state["chroma_weight"] = chroma_weight / total_weight

    def show_options(self):
        with st.sidebar.expander("🛠️ H-Robby 옵션", expanded=False):

            self.reset_chat_button()
            self.embed_documents_button("./manual", "./chroma_db3")
            self.delete_db_button("./chroma_db3")
            self.llm_selector()
            self.model_selector()
            self.temperature_slider()
            self.retriever_weight_sliders()
            st.session_state.setdefault("model", self.MODEL_OPTIONS[0])
            st.session_state.setdefault("temperature", self.TEMPERATURE_DEFAULT_VALUE)
            st.session_state.setdefault("bm25_weight", 0.5)
            st.session_state.setdefault("chroma_weight", 0.5)

    