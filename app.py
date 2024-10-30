import streamlit as st
from document_processor import DocumentProcessor
import pandas as pd

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
            questions = processor.display_questions(num_questions=num_prompts)
            st.write(questions)
            
            # Read the generated CSV file
            output_csv_path = 'out.csv'
            
            # Download button for Output CSV
            with open(output_csv_path, "rb") as file:
                st.download_button("Download Output CSV", file, "out.csv")



if __name__ == "__main__":
    main()
