from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

app = Flask(__name__)

loader = PyPDFLoader("kennis.pdf")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=db.as_retriever()
)

@app.route("/ask", methods=["POST"])
def ask():
    vraag = request.json["question"]
    antwoord = qa.run(vraag)
    return jsonify({"answer": antwoord})

if __name__ == "__main__":
    app.run(port=5000)
