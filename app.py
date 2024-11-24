import streamlit as st
from document_processor import DocumentProcessor
import pandas as pd
import os
import io
import zipfile

# Streamlit App UI and Logic
def main():
    st.title("Batch Document Processor with ZIP Download")

    # PDF Upload
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # Number of Prompts
    num_prompts = st.number_input("Enter the number of prompts to generate", min_value=1, value=3)

    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            # Create a memory buffer for the ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for pdf_file in uploaded_files:
                    with open(pdf_file.name, "wb") as f:
                        f.write(pdf_file.getbuffer())

                    # Process each file
                    processor = DocumentProcessor(pdf_path=pdf_file.name)
                    processor.load_pdf()
                    processor.generate_testset_1(num_questions=num_prompts)

                    # Save outputs
                    base_name = os.path.splitext(pdf_file.name)[0]
                    jsonl_file, csv_file = processor.save_outputs(output_dir=".", base_name=base_name)

                    # Add files to ZIP
                    zipf.write(jsonl_file, os.path.basename(jsonl_file))
                    zipf.write(csv_file, os.path.basename(csv_file))

                    # Clean up temporary files
                    os.remove(jsonl_file)
                    os.remove(csv_file)

            # Finalize the ZIP file
            zip_buffer.seek(0)

            # Provide download button for the ZIP file
            st.download_button(
                label="Download All Processed Files as ZIP",
                data=zip_buffer,
                file_name="processed_files.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()
