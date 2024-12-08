import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from modules.prompt import PROMPT_STR


import langchain
langchain.verbose = False

class Chatbot:

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

    # qa_template = """
    #     You are a helpful AI assistant named H-Robby. The user gives you a file its content is represented by the following pieces of context, use them to answer the question at the end.
    #     If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
    #     If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    #     Use as much detail as possible when responding.

    #     context: {context}
    #     =========
    #     question: {question}
    #     ======
    #     """
    qa_template = PROMPT_STR

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])

    def conversational_chat(self, query):
        
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()

        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=retriever, verbose=True, return_source_documents=True, max_tokens_limit=4097, combine_docs_chain_kwargs={'prompt': self.QA_PROMPT})

        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)
        return result["answer"]

    def ensemble_conversational_chat(self, query, document):

        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        # 첫 번째 리트리버 : 기존 벡터 리트리버
        chroma_retriever = self.vectors.as_retriever()

        # 두 번째 리트리버 : BM25 리트리버
        bm25_retriever = BM25Retriever.from_documents(document)

        # 앙상블 리트리버 생성
        ensemble_retriever = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25_retriever],
            mode="merge",  # 결과 병합
            weight = [st.session_state["chroma_weight"], st.session_state["bm25_weight"]]
        )

        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=ensemble_retriever, verbose=True, return_source_documents=True, max_tokens_limit=4097, combine_docs_chain_kwargs={'prompt': self.QA_PROMPT})

        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)
        return result["answer"]

def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### 토큰 사용량 : {cb.total_tokens} tokens')
    return result 


    
