import os
import uuid
import time
from collections import defaultdict

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, session
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
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))
CORS(app, supports_credentials=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"
DOCS_PATH = "./docs"
CHROMA_PATH = "./chroma_db"

historicos: dict[str, list[str]] = defaultdict(list)

RATE_LIMIT = 20
RATE_WINDOW = 60         
_rate_log: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(session_id: str) -> bool:
    """Retorna True se a requisição for permitida, False se exceder o limite."""
    now = time.time()
    timestamps = _rate_log[session_id]
    
    _rate_log[session_id] = [t for t in timestamps if now - t < RATE_WINDOW]
    if len(_rate_log[session_id]) >= RATE_LIMIT:
        return False
    _rate_log[session_id].append(now)
    return True



def inicializar_conhecimento():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(CHROMA_PATH):
        print("Carregando base vetorial existente...")
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

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
            fontes.add(os.path.basename(fonte))
    return list(fontes)



retriever = inicializar_conhecimento()
llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)

# PROMPT RESTRITIVO: responde apenas com base nos documentos carregados.
# Se a informação não estiver nos documentos, informa ao usuário em vez de
# usar conhecimento geral — importante para um contexto jurídico formal.
SYSTEM_PROMPT = """Você é o Assistente Jurídico Alagoas Inovação, especialista em inovação e tecnologia do estado de Alagoas.

Responda EXCLUSIVAMENTE com base nos trechos da legislação fornecidos no contexto abaixo.
Quando a resposta estiver nos documentos, cite os artigos e leis relevantes.
Se a informação solicitada NÃO estiver nos documentos, responda apenas:
"Não encontrei essa informação na base legislativa carregada. Para questões fora dessa base, consulte um advogado especializado."

Nunca invente informações jurídicas. Nunca use conhecimento externo aos documentos fornecidos.

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
        {
            "context": retriever | formatar_docs,
            "input": RunnablePassthrough(),
            "historico": lambda _: "",  
        }
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    rag_chain = None


def home():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not retriever:
        return jsonify({"error": "Base de conhecimento não carregada. Verifique se há arquivos em ./docs"}), 500

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    session_id = session["session_id"]


    if not check_rate_limit(session_id):
        return jsonify({"error": f"Limite de {RATE_LIMIT} mensagens por minuto atingido. Aguarde um momento."}), 429

    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Mensagem vazia"}), 400

    historico = historicos[session_id]

    try:
        docs_relevantes = retriever.invoke(user_input)
        fontes = extrair_fontes(docs_relevantes)
        contexto = formatar_docs(docs_relevantes)
        historico_str = "\n".join(historico[-6:])

        resposta = (
            prompt
            | llm
            | StrOutputParser()
        ).invoke({
            "context": contexto,
            "input": user_input,
            "historico": historico_str,
        })

        historico.append(f"Usuário: {user_input}")
        historico.append(f"Assistente: {resposta}")
    
        if len(historico) > 10:
            historicos[session_id] = historico[-10:]

        return jsonify({"response": resposta, "fontes": fontes})

    except Exception as e:
        print(f"Erro: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/limpar", methods=["POST"])
def limpar_historico():
    session_id = session.get("session_id")
    if session_id:
        historicos.pop(session_id, None)
    return jsonify({"ok": True})


if __name__ == "__main__":
   
    app.run(host="0.0.0.0", port=8001, debug=False)
