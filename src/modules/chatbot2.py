import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from transformers import pipeline
# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import os
from dotenv import load_dotenv
from modules.prompt import PROMPT_STR, PROMPT_STR_FOR_HF
# from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

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

            # config = GPT2Config.from_pretrained("skt/kogpt2-base-v2")
            self.huggingface_tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            huggingface_model = GPT2LMHeadModel.from_pretrained(self.model_name)

            return huggingface_model
            # return pipeline(
            #     "question-answering", 
            #     model=huggingface_model,
            #     tokenizer=self.huggingface_tokenizer, 
            #     max_new_tokens=150)
            
            # return HuggingFacePipeline(pipeline=pipe)

            # return HuggingFacePipeline.from_model_id(
            #     model_id=self.model_name,  # 사용할 모델의 ID를 지정합니다.
            #     task="text-generation",  # 수행할 작업을 지정합니다. 여기서는 텍스트 생성입니다.
            #     # 파이프라인에 전달할 추가 인자를 설정합니다. 여기서는 생성할 최대 토큰 수를 10으로 제한합니다.
            #     pipeline_kwargs={"max_new_tokens": 512},
            # )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def chat(self, question):

        # ChromaDB에서 context 검색
        # context_data = self.retrieve_context(question)
        documents = self.retriever.invoke(question)
        if not documents:
            # No relevant context found
            return "죄송합니다. 현재 업로드된 정보에 요청하신 정보가 없습니다."
        
        # context 추출
        # context = "\n".join([data["content"] for data in documents])
        context = "\n".join([doc.page_content for doc in documents])

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

            return {"answer": result["answer"], "documents": documents}
        
        elif self.model_type == "Gemini":
            chain = ConversationalRetrievalChain.from_llm(
                llm = self.model,
                retriever = self.retriever,
                verbose = True,
                return_source_documents = True,
                max_tokens_limit=4097,
                combine_docs_chain_kwargs = {"prompt": Chatbot2.QA_PROMPT}
            )
            chain_input = {"question": question, "chat_history": ""}
            result = chain(chain_input)

            # st.session_state["history"].append((question, result["answer"]))
            # count_tokens_chain(chain, chain_input)

            return {"answer": result["answer"], "documents": documents}
        
        elif self.model_type == "HuggingFace":
            
            # LangChain 리트리버를 사용해 관련 문서 검색
            docs = self.retriever.get_relevant_documents(question)
            
            # 검색된 문서를 컨텍스트로 GPT-2 입력 생성
            retrieved_docs = [doc.page_content for doc in docs[:3]]
            context = " ".join(retrieved_docs)
            # input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
            input_text = self.QA_PROMPT_FOR_HF.format(context=context, question=question)

            # GPT-2 답변 생성
            input_ids = self.huggingface_tokenizer.encode(input_text, return_tensors='pt')

            MAX_INPUT_TOKENS = 1000
            # 입력 길이가 초과하면 텍스트 잘라내기
            if len(input_ids[0]) > MAX_INPUT_TOKENS:
                excess_length = len(input_ids[0]) - MAX_INPUT_TOKENS
                # 초과한 길이만큼 context를 잘라냄
                while excess_length > 0 and len(retrieved_docs) > 0:
                    longest_doc = max(retrieved_docs, key=len)
                    retrieved_docs.remove(longest_doc)
                    context = " ".join(retrieved_docs)
                    input_text = self.QA_PROMPT_FOR_HF.format(context=context, question=question)
                    input_ids = self.huggingface_tokenizer.encode(input_text, return_tensors='pt')
                    excess_length = len(input_ids[0]) - MAX_INPUT_TOKENS

            print(f"input_textinput_textinput_textinput_text {input_text}")

            # 최종 답변 생성
            output_ids = self.model.generate(
                input_ids, 
                max_new_tokens=300,  # 생성되는 답변의 최대 토큰 길이
                num_return_sequences=1,
                temperature=0.7,  # 다양성을 줄이고 더 결정적인 답변을 유도
                top_p=0.9,       # 상위 확률의 토큰만 샘플링
                repetition_penalty=1.2  # 반복 생성 방지
            )
            
            generated_text = self.huggingface_tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Answer 부분만 추출
            if "[Answer]" in generated_text:
                return generated_text.split("[Answer]")[1].strip()
            return generated_text.strip()
            
            # if not self.huggingface_tokenizer.pad_token:
            #     self.huggingface_tokenizer.pad_token = self.huggingface_tokenizer.eos_token

            # result = pipeline("question-answering", model="gpt2")
            
            # # result(
            # #     question=question,
            # #     context="장애처리는 1 2 3 순서대로 합니다."
            # # )
            # result(
            #     question="베토벤이 태어난 곳은 어디인가요?",
            #     context="루트비히 판 베토벤은 독일의 서양 고전 음악 작곡가이자 피아니스트이다. 독일의 본에서 태어났으며, 성인이 된 이후 거의 오스트리아 빈에서 살았다. 감기와 폐렴의 합병증으로 투병하다가 57세로 세상을 떠난 그는 고전주의와 낭만주의의 전환기에 활동한 주요 음악가이며, 종종 영웅적인 인물로도 묘사된다. 음악의 성인 즉 악성이라는 별칭으로도 불린다.."
            # )
            # return result

            # # Tokenize and truncate
            # # Ensure the tokenizer has a pad token set
            # if not self.huggingface_tokenizer.pad_token:
            #     self.huggingface_tokenizer.pad_token = self.huggingface_tokenizer.eos_token
            # # if not self.huggingface_tokenizer.eos_token:
            # #     self.huggingface_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # max_length = 150

            # # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # BART 기반 요약 모델
            # # summary = summarizer(context, max_length=500, min_length=100, do_sample=False)[0]["summary_text"]

            # # Tokenize separately and check lengths
            # context_tokens = self.huggingface_tokenizer(context[:150], add_special_tokens=False)["input_ids"]
            # question_tokens = self.huggingface_tokenizer(question, add_special_tokens=False)["input_ids"]

            # print(f"Context token length: {len(context_tokens)}")
            # print(f"Question token length: {len(question_tokens)}")
            # print(f"Total token length: {len(context_tokens) + len(question_tokens)}")

            # inputs = self.huggingface_tokenizer(context[:150] + " " + question, truncation=True, padding='max_length', max_length=150, return_tensors="pt")

            # # print(type(inputs['input_ids']), inputs['input_ids'])
            # context_data = {"context": inputs['input_ids'], "question": question}

            # rag_chain = RunnablePassthrough() | Chatbot2.QA_PROMPT_FOR_HF | self.model | StrOutputParser()
            # result = rag_chain.invoke(context_data)

            # return result

            # chain = ConversationalRetrievalChain.from_llm(
            #     llm = self.model,
            #     retriever = self.retriever,
            #     verbose = True,
            #     # truncation = True,
            #     # return_source_documents = True,
            #     # memory = memory,
            #     # max_length=1000,
            #     max_tokens_limit=1000,
            #     # combine_docs_chain_kwargs = {"prompt": Chatbot2.QA_PROMPT_FOR_HF}
            # )
            # chain_input = {"question": question, "chat_history": ""}
            # result = chain(chain_input)

            # st.session_state["history"].append((question, result["answer"]))
            # count_tokens_chain(chain, chain_input)

            # return {"answer": result["answer"], "documents": documents}

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

    qa_template_for_hf = PROMPT_STR_FOR_HF
    QA_PROMPT_FOR_HF = PromptTemplate(template=qa_template_for_hf, input_variables=["context","question" ])

def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.invoke(query)
        st.write(f'###### 토큰 사용량 : {cb.total_tokens} tokens')
    return result 
