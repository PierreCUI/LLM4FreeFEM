import os
import time
import json

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-5.2"


class ChatGPTCodeWriter:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self._model = ChatOpenAI(
            model=self.model_name,
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            seed=42,
        )

    def initialization(self, prompt_list=None, retriever=None, partial_dict=None):
        self._prompt = ChatPromptTemplate.from_messages(prompt_list)
        if partial_dict:
            self._prompt = self._prompt.partial(**partial_dict)
        if retriever:
            take_question = RunnableLambda(lambda x: x["question"])
            join_docs = RunnableLambda(lambda docs: ". ".join(d.page_content for d in docs))
            self._retriever = RunnableParallel({
                "context": take_question | retriever | join_docs,
                "question": take_question,
            })
        self._parser = StrOutputParser()
        if not retriever:
            self._chain = self._prompt | self._model | self._parser
        else:
            self._chain = self._retriever | self._prompt | self._model | self._parser

    def invoke(self, function_dict):
        return self._chain.invoke(function_dict)


class ChatGPTCodeAgent:
    def __init__(self, history_file=None):
        self.prompt_list = []
        self.retriever = None
        self.history_chat = []
        if not history_file:
            history_file = rf"./history_chat/chat_{time.time()}.json"
        self.history_file = history_file
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        self.writer = ChatGPTCodeWriter()

    def invoke(self, function_dict, partial_dict=None):
        self.writer.initialization(
            self.prompt_list,
            retriever=self.retriever,
            partial_dict=partial_dict,
        )
        full_text = self.writer.invoke(function_dict)
        self.history_chat.append({"input": self.prompt_list, "output": full_text})
        self.save_history()
        return full_text

    def set_prompt_list(self, prompt_list):
        self.prompt_list = prompt_list

    def set_retriever(self, retriever):
        self.retriever = retriever

    def set_retriever_from_documents(self, model_name, documents_list, search_type, search_kwargs):
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        docs = []
        for path in documents_list:
            loaded_docs = TextLoader(path, encoding="utf-8").load()
            for doc in loaded_docs:
                docs.append(doc)
        vector_store_db = FAISS.from_documents(docs, embedding=embeddings)
        retriever = vector_store_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        self.set_retriever(retriever)

    def load_history(self, history_file):
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                try:
                    self.history_chat = json.load(f)
                except json.JSONDecodeError:
                    pass
        history_chat_list = []
        for record in self.history_chat:
            for (person, message) in record["input"]:
                history_chat_list.append((person, message))
            history_chat_list.append(("assistant", record["output"]))
        return history_chat_list

    def save_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history_chat, f, ensure_ascii=False, indent=4)
