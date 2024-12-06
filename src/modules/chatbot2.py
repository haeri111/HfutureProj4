import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from transformers import pipeline
# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv
from modules.prompt import PROMPT_STR
# from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

import langchain
# langchain.verbose = False

# Load environment variables
load_dotenv()

class Chatbot2:

    def __init__(self, model_type, model_name, temperature, retriever):
        self.model_type = model_type
        self.model_name = model_name
        self.temperature = temperature
        self.retriever = retriever
        self.model = self.initialize_model()
    
    def initialize_model(self):
        # model_type 파라미터를 기반으로 LLM 모델 초기화
        if self.model_type == "OpenAI":
            return ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        
        elif self.model_type == "Gemini":
            return ChatGoogleGenerativeAI(model=self.model_name, temperature=self.temperature)

        elif self.model_type == "HuggingFace":
            # 참고: huggingface_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            # huggingface_model = GPT2LMHeadModel.from_pretrained("gpt2")

            config = GPT2Config.from_pretrained("skt/kogpt2-base-v2")
            self.huggingface_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            huggingface_model = GPT2LMHeadModel.from_pretrained("gpt2")

            return pipeline(
                "question-answering",
                model = self.model_name,
                tokenizer = self.model_name
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def chat(self, question):

        # ChromaDB에서 context 검색
        context = self.retrieve_context(question)
        if not context:
            # No relevant context found
            return "죄송합니다. 현재 업로드된 정보에 요청하신 정보가 없습니다."

        # llm 분기처리
        if self.model_type == "OpenAI":
            chain = ConversationalRetrievalChain.from_llm(
                llm = self.model,
                retriever = self.retriever,
                verbose = True,
                return_source_documents = True,
                max_tokens_limit=4097,
                combine_docs_chain_kwargs = {"prompt": Chatbot2.QA_PROMPT}
            )
            chain_input = {"question": question, "chat_history": st.session_state["history"]}
            result = chain(chain_input)

            st.session_state["history"].append((question, result["answer"]))
            count_tokens_chain(chain, chain_input)
            return result["answer"]
        
        elif self.model_type == "Gemini":
            # Gemini 대화
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.model,
                retriever=self.retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": Chatbot2.QA_PROMPT},
            )
            chain_input = {"question": question, "chat_history": st.session_state.get("history", [])}
            result = chain(chain_input)

            # Update session history
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].append((question, result["answer"]))

            return result["answer"]
            # chat = self.model.start_chat(history=[])
            # chat.history.append({"text": context, "role": "system"})
            # response = self.model.start_chat(prompt=f"{context}\n\n{question}")
            # return response
        
        elif self.model_type == "HuggingFace":
            # Hugging Face QA 모델 대화
            inputs = self.huggingface_tokenizer(question, return_tensors="pt")
            # response = self.model({"question": question, "context": context})
            outputs = self.model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
            self.huggingface_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response["answer"]

    def retrieve_context(self, question):
        # ChromaDB 기반 질의 검색
        if self.retriever:
            documents = self.retriever.invoke(question)
            if not documents:
                print(f"No documents found for query: {question}")
            else:
                print(f"Documents retrieved: {[doc.page_content for doc in documents]}")

            context = "\n".join([doc.page_content for doc in documents])
            return context
        else:
            return None
        
    qa_template = PROMPT_STR
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])


def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 
