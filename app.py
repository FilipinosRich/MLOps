import os
import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

model_id = "openai-community/gpt2-medium"
conv_model = HuggingFaceHub(huggingfacehub_api_token=
                            os.environ['HUGGINGFACEHUB_API_TOKEN'],
                            repo_id=model_id,
                            model_kwargs={"max_new_tokens": 150})

add_llm_provider(
    LangchainGenericProvider(
        # It is important that the id of the provider matches the _llm_type
        id=conv_model._llm_type,
        # The name is not important. It will be displayed in the UI.
        name="HuggingFaceHub",
        # This should always be a Langchain llm instance (correctly configured)
        llm=conv_model,
        # If the LLM works with messages, set this to True
        is_chat=False
    )
)


template = """My name is {query} and I am"""

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=['query'])
    conv_chain = LLMChain(llm=conv_model,
                          prompt=prompt,
                          verbose=True)

    cl.user_session.set("llm_chain", conv_chain)

@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")

    # Assuming 'message' is already the correct text to be processed
    # If 'message' is a Message object, we need to extract the text or content from it
    if isinstance(message, cl.message.Message):
        message_text = message.content  # Adjust this according to the actual structure of your Message class
    else:
        message_text = message  # It's already a string

    res = await llm_chain.acall(message_text, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Prepare the response text
    response_text = res.get("text", "Error processing your request")
    await cl.Message(content=response_text).send()