import gradio as gr

# Invoke fintune model
def generate_bot_reply(message:str, history: list[dict])-> str:
    """
    message: the latest user message
    history: previous chat history in [[user, assistant], ...] format

    Return:
        assistant response as a string
    """
    return f"You said {message}"
    pass

# Chatbot logic function
def user_submit(message: str, history: list[dict]):
    """
    Triggered when the user sends messages
    """
    if message.strip():

        reply = generate_bot_reply(message, history)
        updated_history = history + [
            {"role":"user", "content": message},
            {"role":"assistant", "content": reply }
        ]
    return "", updated_history, updated_history

def clear_chat_history():
    return [], []

# UI part
with gr.Blocks(title="Chatbot") as demo:
    gr.Markdown("# Faithful Q&A System Demo")
    gr.Markdown("Type a message below to start the chat.")

    chatbot = gr.Chatbot(label="Chat", height=500)
    # chatbot = gr.Chatbot(type="messages", height=450)
    message = gr.Textbox(placeholder="Enter your message..", scale=8)
    with gr.Row():
        send_button = gr.Button("Send", variant="primary", scale=1)
        clear_button = gr.Button("Clear")

    state = gr.State([])

    send_button.click(user_submit, [message, state], [message, chatbot, state])
    message.submit(user_submit, [message, state], [message, chatbot, state])
    clear_button.click(clear_chat_history, [], [chatbot, state])
    pass

demo.launch()