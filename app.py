import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

os.environ['OPENAI_API_KEY'] = 'input api key here'

llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

loader = PyPDFLoader('agrana.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')

vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('JARED DUNN AI 📄🦜')
prompt = st.text_input('Enter your prompt here')

if prompt:
    response = agent_executor.run(prompt)
    st.write(response)
    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt) 
        st.write(search[0][0].page_content) 