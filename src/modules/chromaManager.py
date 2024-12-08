from typing import Optional
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
# from modules.myembedding import MyEmbeddingFunction
import streamlit as st
import openai

class ChromaManager:
    def __init__(self, persist_directory: str):
        """
        ChromaManager 초기화

        :param persist_directory: 로컬 디렉토리 경로
        :param embedding_function: 사용될 임베딩 함수 (default: OpenAIEmbeddings)
        """
        self.persist_directory = persist_directory
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        # self.embedding_function = MyEmbeddingFunction()
        self.collections = {} # collections 속성 초기화
        self.client = PersistentClient(path=persist_directory)

        # Chroma DB 로드
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_function,
        )

        # self.vector_collection = self.client.get_or_create_collection("vector_table", embedding_function=self.embedding_function)
        # self.retriever = self.vector_collection.as_retriever(search_kwargs={"k": 10})
        self.retriever = self.db.as_retriever(search_kwargs={"k": 5})

        # 세션에 db, retriever 저장
        st.session_state["db"] = self.db
        st.session_state["retriever"] = self.retriever
        st.session_state["client"] = self.client
        # else:
        #     # 세션에서 db, retriever 가져오기
        #     self.db = st.session_state["db"]
        #     self.retriever = st.session_state["retriever"]
        #     self.client = st.session_state["client"]

    def get_or_create_collection(self, name: str) -> Optional[object]:
        """
        컬렉션을 가져오거나 존재하지 않으면 생성.

        :param name: 컬렉션 이름
        :return: 컬렉션 객체
        """
        if name in self.collections:
            print(f"Collection '{name}' already exists. Returning existing collection.")
            return self.collections[name]

        # 컬렉션 생성 및 UNIQUE 제약 조건 방지
        try:
            collection = self.client.get_or_create_collection(name=name)
            self.collections[name] = collection
            print(f"Collection '{name}' created successfully.")
            return collection
        except Exception as e:
            print(f"Error creating collection '{name}': {e}")
            return None

    def initialize_collections(self, collection_names: list):
        """
        컬렉션 초기화

        :param collection_names: 초기화할 컬렉션 이름 목록
        """
        for name in collection_names:
            self.get_or_create_collection(name)

    def insert_into_collection(self, collection_name: str, documents: list):
        """컬렉션에 문서 삽입"""
        collection = self.get_or_create_collection(collection_name)
        if not collection:
            print(f"Collection '{collection_name}' not found. Cannot insert data.")
            return
        try:
            for document in documents:
                collection.add(**document)
            print(f"Inserted {len(documents)} documents into '{collection_name}' collection.")
        except Exception as e:
            print(f"Error inserting documents into '{collection_name}': {e}")

    def query_collection(self, collection_name: str, query_text: str, n_results: int = 5):
        """컬렉션에서 쿼리 실행"""
        collection = self.get_or_create_collection(collection_name)
        if not collection:
            print(f"Collection '{collection_name}' not found. Cannot query.")
            return
        
        try:
            results = collection.query(query_texts=[query_text], n_results=n_results)
            print(f"Query Results from '{collection_name}':")
            for idx, result in enumerate(results.get("documents", [])):
                print(f"Result {idx + 1}: {result}")
        except Exception as e:
            print(f"Error querying '{collection_name}': {e}")
            
    def get_retriever(self):
        """
        검색 리트리버 반환

        :return: Chroma 리트리버
        """
        if not self.retriever:
            print("Retriever is not initialized.")
        else:
            print("Retriever initialized successfully.")
        return self.retriever
    

    def debug_collections(self):
        for name, collection in self.collections.items():
            print(f"*********************Collection: {name}, Documents: {collection.count()}")
                
            