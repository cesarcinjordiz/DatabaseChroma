from huggingface_hub import InferenceClient
from langchain.prompts import ChatPromptTemplate

# Configurar el cliente de Hugging Face
client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token="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"
)

# Prompt para la generación de respuesta
template = """Debes de responder a cualquier pregunta:
Pregunta: {question}
"""

prompt_template = ChatPromptTemplate.from_template(template)

def main():
    while True:
        texto_introducido = input("Cesar: ")
        if texto_introducido.lower() == 'adios':
            print('¡Hasta luego!')
            break
        
        # Crear el prompt usando el template
        prompt = prompt_template.format(question=texto_introducido)
        
        # Realizar la inferencia usando el cliente de Hugging Face
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(messages=messages, max_tokens=500)
        
        # Imprimir el resultado
        if hasattr(response, 'choices') and response.choices:
            generated_text = response.choices[0].message.content
        else:
            generated_text = str(response)  # Si la respuesta es un string
        
        print(generated_text)

if __name__ == "__main__":
    main()
