import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from pdf_processing import load_pdf_and_extract_text
from vector_store import get_or_create_vector_store
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

OpenAI.api_key = api_key

def main():
    # Sidebar contents
    with st.sidebar:
        st.title('🤗💬 LLM Chat App')
        add_vertical_space(5)

    st.header("Chat with pdf")
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf:
        text = load_pdf_and_extract_text(pdf)
        store_name = pdf.name[:-4]
        VectorStore = get_or_create_vector_store(text, store_name)

        query = st.text_input("Ask questions about your PDF file:")
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)


if __name__ == '__main__':
    load_dotenv()
    main()
