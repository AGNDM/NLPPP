import gradio as gr

# Invoke fintune model
def generate_bot_reply(message:str, history: list[list[str]])-> str:
    """
    message: the latest user message
    history: previous chat history in [[user, assistant], ...] format

    Return:
        assistant response as a string
    """
    return f"You said {message}"
    pass

# Chatbot logic function
def chat(message: str, history: list[list[list]]):
    """
    Triggered when the user sends messages
    """
    if message.strip():
        
        reply = generate_bot_reply(message, history)
        history = history + [[message, reply]]

    return "", history