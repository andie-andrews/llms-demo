'''
Gradio UI on top of ollama chatbot
'''

import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = 'qwen2.5:3b'
temperature = 0.9

system_prompt = (
    'You are a helpful teaching assistant at an AI/ML boot camp.',
    'Answer questions in simple language with examples when possible.',
    'Answer in the style of a pirate and use nautical themed analogies.'
)

# Initialize the Ollama chatbot
llm = ChatOllama(
    model=model,
    temperature=temperature
)

def respond(message, history):
    '''Sends message to model, gets response back'''

    messages = [SystemMessage(content=system_prompt)]

    for user_msg, assistant_msg in history:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=assistant_msg))

    messages.append(HumanMessage(content=message))
    response = llm.invoke(messages)

    return response.content

demo = gr.ChatInterface(
    fn=respond,
    title=f'Chatbot: {model}'
)

if __name__ == '__main__':

    demo.launch()