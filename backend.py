import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context just say
    "answer is not available in the context". Don't make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )
    return chain

def answer_question(user_question: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # IMPORTANT: allow_dangerous_deserialization=True new LangChain versions mein zaroori ho sakta hai
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("\nRunning similarity search...")
    docs = new_db.similarity_search(user_question, k=3)

    print("\nTop documents:")
    for i, d in enumerate(docs, start=1):
        print(f"\n----- DOC {i} -----")
        print(d.page_content[:500])

    chain = get_conversational_chain()

    print("\nGetting final answer from Gemini...\n")
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print("=== ANSWER ===")
    print(response["output_text"])

if __name__ == "__main__":
    while True:
        q = input("\nEnter your question (or 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break
        answer_question(q)
