from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

local_path = "datoscasa2.txt"

# Local .txt file uploads
if local_path:
    loader = TextLoader(file_path=local_path)  # Use TextFileLoader for .txt
    data = loader.load()  # Load with explicit encoding
else:
    print("Upload a .txt file")
# Split and chunk 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Define a persistent collection name (avoid deletion after each interaction)
collection_name = "local-rag"
custom_db_directory = "/Users/cesarhernandez/Documents/PlatformIO/Projects/RAG-1/prueba2"

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="phi3", show_progress=False),
    collection_name=collection_name,
    persist_directory=custom_db_directory
)

# LLM from Ollama
local_model = "phi3"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Eres una asistente y te llamas Lara. 
    Pregunta original: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Debes de responder a cualquier pregunta:
{context}
Pregunta: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def main():
  while True:
    texto_introducido=input("Cesar: ")
    if texto_introducido.lower() == 'adios':
        print('¡Hasta luego!')
        break
    resultado = chain.invoke(texto_introducido)
    print(resultado)
    
if __name__ == "__main__":
    main()
