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

def clear_chat_history():
    return []

# UI part
with gr.Blocks(title="Chatbot") as demo:
    gr.Markdown("# Faithful Q&A system Demo")
    gr.Markdown("Type a message below to chat with faithful Q&A system.")

    chatbot = gr.Chatbot(label="Chat", height=500)
    message = gr.Textbox(
        label="Your message",
        placeholder="Ask something...",
        lines=2
    )

    with gr.Row():
        send_button = gr.Button("Send", variant="primary")
        clear_button = gr.Button("Clear")

    state = gr.State([])

    send_button.click(
        fn=chat,
        inputs=[message, state],
        outputs=[message, chatbot],
    ).then(
        fn=lambda history: history,
        inputs=[chatbot],
        outputs=[state],
    )

    message.submit(
        fn=chat,
        inputs=[message, state],
        outputs=[message, chatbot],
    ).then(
        fn=lambda history: history,
        inputs=[chatbot],
        outputs=[state],
    )

    clear_button.click(
        fn=clear_chat_history,
        inputs=[],
        outputs=[chatbot],
    ).then(
        fn=lambda: [],
        inputs=[],
        outputs=[state],
    )
    pass

demo.launch()