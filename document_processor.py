import os
import zipfile
import io
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from giskard.rag import KnowledgeBase, generate_testset
import json
from openai import OpenAI
import giskard


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY")
llm_client=giskard.llm.set_llm_model("gpt-3.5-turbo-0125")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("API_KEY"))

class EmbeddingWrapper:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def embed(self, texts):
        # Use `embed_documents` or `embed_query` depending on your use case
        return self.embedding_model.embed_documents(texts)
embedding_wrapper = EmbeddingWrapper(embedding_model)
class DocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        # self.openai_api_key = OPENAI_API_KEY
        self.model = embedding_model
        self.documents = []
        self.vectorstore = None
        self.knowledge_base = None
        self.testset = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)

    def load_pdf(self):
        """Load PDF and split it into documents."""
        loader = PyPDFLoader(self.pdf_path)
        self.documents = loader.load_and_split(self.text_splitter)
        print(f"Loaded {len(self.documents)} documents from {self.pdf_path}.")

    def generate_testset_1(self, num_questions=15, 
                           description="You are a Legal expert creating in-depth and context-rich questions and "
    "answers from the document content. Ensure the questions encourage analytical "
    "thinking and provide detailed, well-explained answers, including examples where possible."):
        """Generate a test set based on the knowledge base."""
        self.knowledge_base = KnowledgeBase(pd.DataFrame([d.page_content for d in self.documents], columns=["text"]),
                                            embedding_model=embedding_wrapper,llm_client=llm_client)
        self.testset = generate_testset(
            self.knowledge_base,
            num_questions=num_questions,
            agent_description=description)
        print(f"Generated testset with {num_questions} questions.")

    def save_outputs(self, output_dir, base_name):
        """Save testset and output DataFrame to files."""
        # Save JSONL
        jsonl_file = os.path.join(output_dir, f"{base_name}.jsonl")
        self.testset.save(jsonl_file)
        # Save CSV
        test_set_df = self.testset.to_pandas()
        csv_file = os.path.join(output_dir, f"{base_name}.csv")
        test_set_df.to_csv(csv_file, index=False)
        return jsonl_file, csv_file

    def display_questions(self, num_questions=3):
        """Print the specified number of questions and answers."""
        test_set_df = self.testset.to_pandas()
        for index, row in enumerate(test_set_df.head(num_questions).iterrows()):
            print(f"Question {index + 1}: {row[1]['question']}")
            print(f"Reference answer: {row[1]['reference_answer']}")
            print("Reference context:")
            print(row[1]['reference_context'])
            print("******************", end="\n\n")

