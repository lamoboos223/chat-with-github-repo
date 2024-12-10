import chainlit as cl
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

@cl.on_chat_start
def init():
    # Initialize the database
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Store the database in the user session
    cl.user_session.set("db", db)

@cl.on_message
async def main(message: cl.Message):
    # Get the database from the user session
    db = cl.user_session.get("db")
    
    # Search the DB
    results = db.similarity_search_with_score(message.content, k=5)
    
    # Create context from results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Format prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=message.content)
    
    # Show the prompt to help with debugging
    await cl.Message(
        content=f"Searching database with context...",
        author="System"
    ).send()
    
    # Get response from model
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)
    
    # Get sources
    sources = [doc.metadata.get("source", "No source provided") for doc, _score in results]
    
    # Send the response
    await cl.Message(content=response_text).send()
    
    # Send the sources as a separate message
    source_message = "Sources:\n" + "\n".join([f"- {source}" for source in sources])
    await cl.Message(content=source_message, author="System").send()