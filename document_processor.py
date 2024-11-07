import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from giskard.rag import KnowledgeBase, generate_testset
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY")

class DocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.openai_api_key = OPENAI_API_KEY
        self.model = "gpt-4o-mini"
        self.documents = []
        self.vectorstore = None
        self.knowledge_base = None
        self.testset = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

    def load_pdf(self):
        """Load PDF and split it into documents."""
        loader = PyPDFLoader(self.pdf_path)
        self.documents = loader.load_and_split(self.text_splitter)
        print(f"Loaded {len(self.documents)} documents from {self.pdf_path}.")

    def create_vectorstore(self):
        """Create a vector store from the loaded documents."""
        self.vectorstore = DocArrayInMemorySearch.from_documents(
            self.documents, 
            embedding=OpenAIEmbeddings()
        )

    def generate_testset(self, num_questions=15, description="A chatbot answering questions about the Legal Payment of Taxes"):
        """Generate a test set based on the knowledge base."""
        self.knowledge_base = KnowledgeBase(pd.DataFrame([d.page_content for d in self.documents], columns=["text"]))
        self.testset = generate_testset(
            self.knowledge_base,
            num_questions=num_questions,
            agent_description=description
        )
        print(f"Generated testset with {num_questions} questions.")

    def save_outputs(self):
        """Save testset and output DataFrame to files, including IDs and context."""
        
        # Save the testset to a JSONL file first
        self.testset.save("testset.jsonl")
         
        # Convert the testset to a DataFrame
        test_set_df = self.testset.to_pandas()

        test_set_df.to_csv('out.csv', index=False)
        print("Saved testset and output DataFrame to 'testset.jsonl' and 'out.csv'.")



    def display_questions(self, num_questions=3):
        """Print the specified number of questions and answers."""
        test_set_df = self.testset.to_pandas()
        for index, row in enumerate(test_set_df.head(num_questions).iterrows()):
            print(f"Question {index + 1}: {row[1]['question']}")
            print(f"Reference answer: {row[1]['reference_answer']}")
            print("Reference context:")
            print(row[1]['reference_context'])
            print("******************", end="\n\n")

# Streamlit App UI and Logic
def main():
    st.title("Document Processor with OpenAI Test Set Generator")
   
    # PDF Upload
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    # Number of Prompts
    num_prompts = st.number_input("Enter the number of prompts to generate", min_value=1, value=3)
    
    if st.button("Process Document"):
        if not pdf_file:
            st.warning("Please upload a PDF file.")
        else:
            # Save the uploaded file temporarily
            with open(pdf_file.name, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Instantiate DocumentProcessor without API key input
            processor = DocumentProcessor(pdf_path=pdf_file.name)
            
            processor.load_pdf()
            processor.create_vectorstore()
            processor.generate_testset(num_questions=num_prompts)
            processor.save_outputs()
            
            # Display Questions
            st.write("Generated Questions:")
            processor.display_questions(num_questions=num_prompts)  # This will print to the console; adapt as needed.
            
            # Download options
            st.download_button("Download Test Set JSONL", "testset.jsonl", "testset.jsonl", mime="application/json")
            st.download_button("Download Output CSV", "out.csv", "out.csv", mime="text/csv")

if __name__ == "__main__":
    main()
