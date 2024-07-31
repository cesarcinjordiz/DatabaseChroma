from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LLM from Ollama
local_model = "phi3"
llm = ChatOllama(model=local_model)

# Prompt para la generación de respuesta
template = """Debes de responder a cualquier pregunta:
Pregunta: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Configurar la cadena para el procesamiento
chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def main():
    while True:
        texto_introducido = input("Cesar: ")
        if texto_introducido.lower() == 'adios':
            print('¡Hasta luego!')
            break
        resultado = chain.invoke(texto_introducido)
        print(resultado)

if __name__ == "__main__":
    main()
