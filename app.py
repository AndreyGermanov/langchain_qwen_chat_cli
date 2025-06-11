from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from qwen_chat import LocalQwenChat
import torch

# Function loads content of all articles in "text" folder to Chroma vector storage
# and returns "retriever" object from it 
def get_retriever():
    loader = DirectoryLoader("texts")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,        
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k":10})
    return retriever

# Helper function to concatenate documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Executes query on Qwen Chat, using 10 more releavant chunks from vector storage
# as a context and returns the answer
def request_qwen(model,query):

    prompt_template = """
    Answer the question based only on the following context.
If the context doesn't contain the answer, say that you can't answer and ask to provide more context.

Context:
{context}

Question: {question}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    rag_chain = (
        {"context": get_retriever() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
    )

    response = rag_chain.invoke(query)
    return response.content

# Clean GPU cache
torch.cuda.empty_cache()

model = LocalQwenChat()

# Main loop: asks user for queries until receives "exit"
while True:
    query = input("Enter query: ")
    if query == "exit":
        break
    print(request_qwen(model,query))  