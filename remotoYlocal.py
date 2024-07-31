from huggingface_hub import InferenceClient
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import contextlib
import io

# Configurar el cliente de Hugging Face
client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token="hf_IKVbAPjJukFCLloEKeYdupabzOvIfXqni"
)

# Configurar el modelo local (Phi-3)
local_model = "phi3"
llm = ChatOllama(model=local_model)

# Prompt para la generación de respuesta
template = """Debes de responder a cualquier pregunta:
Pregunta: {question}
"""

prompt_template = ChatPromptTemplate.from_template(template)

# Configurar la cadena para el procesamiento local
chain = (
    {"question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# Context manager para suprimir la salida estándar y la salida de error
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(io.StringIO(), 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield

def main():
    while True:
        texto_introducido = input("Cesar: ")
        if texto_introducido.lower() == 'adios':
            print('¡Hasta luego!')
            break
        
        # Crear el prompt usando el template
        prompt = prompt_template.format(question=texto_introducido)
        
        try:
            # Suprimir la salida durante la inferencia
            with suppress_stdout_stderr():
                # Realizar la inferencia usando el cliente de Hugging Face
                messages = [{"role": "user", "content": prompt}]
                response = client.chat_completion(messages=messages, max_tokens=500)
            
            # Imprimir el resultado
            if hasattr(response, 'choices') and response.choices:
                generated_text = response.choices[0].message.content
            else:
                generated_text = str(response)  # Si la respuesta es un string
            
            print(generated_text)
        except Exception as e:
            # Si hay un error, suprimir el mensaje y usar el modelo local
            error_message = str(e)
            if "Bad request" in error_message or "does not seem to support chat completion" in error_message:
                pass  # Suprimir mensaje específico
            try:
                resultado = chain.invoke({"question": texto_introducido})
                print(resultado)
            except Exception as local_e:
                print(f"Error al usar el modelo local: {local_e}")

if __name__ == "__main__":
    main()
