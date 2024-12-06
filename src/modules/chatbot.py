import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from transformers import pipeline
# from langchain_google_genai import ChatGoogleGenerativeAI

import langchain
langchain.verbose = False

class Chatbot:

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

        # Hugging Face 모델 초기화
        self.huggingface_pipeline = self.load_huggingface_pipeline()

    @staticmethod
    @st.cache_resource
    def load_huggingface_pipeline():
        """
        Load Hugging Face DistilBERT model for question-answering.
        """
        return pipeline(
            "question-answering",
            model="distilbert-base-uncased-distilled-squad",  # DistilBERT QA 모델
            tokenizer="distilbert-base-uncased-distilled-squad",
        )

    qa_template = """
        You are a helpful AI assistant named Robby. The user gives you a file its content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
        Use as much detail as possible when responding.

        context: {context}
        =========
        question: {question}
        ======
        """

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()

        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=retriever, verbose=True, return_source_documents=True, max_tokens_limit=4097, combine_docs_chain_kwargs={'prompt': self.QA_PROMPT})

        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)
        return result["answer"]

    def huggingface_chat(self, question, context):
        """
        Start a conversational chat with Hugging Face DistilBERT model.
        """
        response = self.huggingface_pipeline({"question": question, "context": context})
        return response["answer"]

    def gemini_chat(self, question, context):
        """
        Start a conversational chat with Hugging Face DistilBERT model.
        """
        response = self.huggingface_pipeline({"question": question, "context": context})
        return response["answer"]

def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 


    
