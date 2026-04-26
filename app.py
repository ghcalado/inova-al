import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"
DOCS_PATH = "./docs"
CHROMA_PATH = "./chroma_db"

# Histórico de conversa (memória da sessão)
historico = []


def inicializar_conhecimento():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Se já existe base vetorial salva, carrega direto (sem reprocessar)
    if os.path.exists(CHROMA_PATH):
        print("Carregando base vetorial existente...")
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    # Primeira vez: processa os documentos e salva
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        print(f"Pasta '{DOCS_PATH}' criada. Adicione seus arquivos lá.")
        return None

    pdf_loader = DirectoryLoader(DOCS_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DOCS_PATH, glob="./*.txt", loader_cls=TextLoader)
    docs = pdf_loader.load() + txt_loader.load()

    if not docs:
        print("Aviso: Nenhum documento encontrado em ./docs")
        return None

    print(f"{len(docs)} páginas encontradas. Processando...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("Base vetorial criada e salva!")
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def formatar_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def extrair_fontes(docs):
    fontes = set()
    for doc in docs:
        fonte = doc.metadata.get("source", "")
        if fonte:
            nome = os.path.basename(fonte)
            fontes.add(nome)
    return list(fontes)


# Inicialização
retriever = inicializar_conhecimento()
llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)

SYSTEM_PROMPT = """Você é o Assistente Jurídico Alagoas Inovação, especialista em inovação e tecnologia.

Use os trechos da legislação fornecidos como sua principal fonte de informação.
Quando a resposta estiver nos documentos, cite os artigos relevantes.
Quando não estiver nos documentos, responda com seu conhecimento geral sobre direito e inovação, 
deixando claro que a informação não vem da legislação local.

Contexto dos documentos:
{context}

Histórico da conversa:
{historico}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

if retriever:
    rag_chain = (
        {"context": retriever | formatar_docs, "input": RunnablePassthrough(), "historico": lambda _: "\n".join(historico[-6:])}
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    rag_chain = None


@app.route("/")
def home():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not rag_chain:
        return jsonify({"error": "Base de conhecimento não carregada. Verifique se há arquivos em ./docs"}), 500

    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Mensagem vazia"}), 400

    try:
        # Busca os documentos relevantes para pegar as fontes
        docs_relevantes = retriever.invoke(user_input)
        fontes = extrair_fontes(docs_relevantes)

        # Gera a resposta
        resposta = rag_chain.invoke(user_input)

        # Salva no histórico
        historico.append(f"Usuário: {user_input}")
        historico.append(f"Assistente: {resposta}")

        # Mantém só as últimas 10 mensagens no histórico
        if len(historico) > 10:
            historico.pop(0)
            historico.pop(0)

        return jsonify({
            "response": resposta,
            "fontes": fontes
        })

    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/limpar", methods=["POST"])
def limpar_historico():
    historico.clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)