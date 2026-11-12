import json
import logging
import os
import re
import tempfile

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langserve import RemoteRunnable

log = logging.getLogger(__name__)

DEFAULT_LLM_URL = "http://localhost:8000/llm/"
try:
    LLM_URL = st.secrets.get("LLM_URL", DEFAULT_LLM_URL)
except Exception:
    LLM_URL = DEFAULT_LLM_URL

GREETING = "안녕하세요."

HEADER = "아래 [참고 정보]와 [이전 대화]를 참고해서 [질문]에 짧게 답해 줘."
UNIFIED = HEADER + "\n\n[참고 정보]\n{context}\n\n[이전 대화]\n{history}\n\n[질문]\n{question}"

KEY_LABELS = {
    "name": "이름", "address": "집 주소", "phone": "전화번호",
    "school": "학교", "major": "전공", "year": "학년",
    "student_id": "학번", "class": "반", "family": "가족",
}
LEAK_MARKERS = ["Below is an instruction", "###", "<s>", "</s>", "[참고 정보]", "[이전 대화]", "[질문]"]
SMALL_DOC_CHARS = 2000   
HISTORY_TURNS = 3
MAX_TURN_CHARS = 60


# 정ㅂ보처리ㄹ
IDENTITY_MODE = "rag"
MODEL_NAME = "llm3"   
BASE_MODEL = "Llama-3-Open-Ko-8B(Llama 3 기반 8B 한국어 모델)"

MODEL_INFO_DOC = (
    f"모델 이름: {MODEL_NAME}\n"
    f"LLM 모델: {BASE_MODEL}\n"
    "LLM 버전: Llama 3 (8B)\n"
    "학습 방법: LoRA(PEFT) 맞춤 학습 (Colab T4, Unsloth)\n"
    "실행 방식: Ollama로 내 컴퓨터에서 직접 실행 (외부 API 사용 안 함)"
)
FIXED = {
    "version": f"나는 {MODEL_NAME}이야. 기반 모델은 {BASE_MODEL}이야.",
    "intro": (f"나는 {BASE_MODEL}를 LoRA(PEFT)로 맞춤 학습한 {MODEL_NAME}이야. "
              "Gemini 같은 외부 API 없이 Ollama로 내 컴퓨터에서 직접 실행되고 있어."),
    "train": ("Colab(T4)에서 Unsloth로 LoRA(r=16) 학습을 했어. 학과/개인 Q&A 데이터를 7에포크 학습하고, "
              "q5_k_m GGUF로 변환해서 Ollama에 등록했어."),
}
BOT_WORDS = ["너", "넌", "니가", "네가", "당신"]


def identity_kind(question):
    """모델 이름/버전/학습 방법을 묻는 질문이면 종류를, 아니면 None"""
    if IDENTITY_MODE == "off":
        return None
    t = question.lower().replace(" ", "")
    about_bot = any(w in t for w in BOT_WORDS) or any(k in t for k in ["어떤모델", "무슨모델", "어떤llm", "무슨llm"])
    if "버전" in t and ("llm" in t or "모델" in t or "ai" in t or about_bot):
        return "version"
    if about_bot and any(k in t for k in ["어떻게학습", "어떻게만들", "어떻게훈련", "파인튜닝", "학습방법", "학습시켰", "만들었"]):
        return "train"
    if about_bot and any(k in t for k in ["누구", "정체", "모델", "llm", "ai"]):
        return "intro"
    return None


def clean_answer(text):
    for m in LEAK_MARKERS:
        if m in text:
            text = text.split(m)[0]
    return text.strip()


def trunc(s, n=MAX_TURN_CHARS):
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


# 문서 처리
def get_text(files):
    docs = []
    for f in files:
        name = f.name.lower()
        if name.endswith(".json"):
            data = json.loads(f.getvalue().decode("utf-8"))
            if isinstance(data, dict):
                text = "\n".join(f"{KEY_LABELS.get(k, k)}: {v}" for k, v in data.items())
                m = re.search(r"우편번호\s*(\d{5})", str(data.get("address", "")))
                if m:
                    text += f"\n우편번호: {m.group(1)}"
            else:
                text = json.dumps(data, ensure_ascii=False)
            docs.append(Document(page_content=text, metadata={"source": f.name}))
        elif name.endswith(".txt"):
            docs.append(Document(page_content=f.getvalue().decode("utf-8"), metadata={"source": f.name}))
        elif name.endswith((".pdf", ".docx")):
            suffix = ".pdf" if name.endswith(".pdf") else ".docx"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.getvalue())
                path = tmp.name
            if suffix == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(path)
            else:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(path)
            docs.extend(loader.load_and_split())
            os.remove(path)
        log.info("Loaded %s", f.name)
    return docs


def build_retriever(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80).split_documents(docs)
    emb = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, emb).as_retriever(search_type="mmr", search_kwargs={"k": 3})


def get_context(question):
    if st.session_state.doc_text:
        return st.session_state.doc_text[:1500]
    if st.session_state.retriever is not None:
        docs = st.session_state.retriever.invoke(question)
        return "\n\n".join(d.page_content for d in docs)[:1200]
    return "없음"


def get_history():
    msgs = st.session_state.messages
    turns = []
    for i in range(0, len(msgs) - 1, 2):
        if msgs[i].role == "user" and msgs[i + 1].role == "assistant":
            turns.append((trunc(msgs[i].content), trunc(msgs[i + 1].content)))
    turns = turns[-HISTORY_TURNS:]
    if not turns:
        return "없음"
    return "\n".join(f"사용자: {q}\n챗봇: {a}" for q, a in turns)


def main():
    st.set_page_config(page_title="jyg 맞춤형 챗봇", page_icon=":books:")
    st.title("jyg 맞춤형 LLM 챗봇 :books:")

    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("doc_text", None)
    ss.setdefault("retriever", None)
    ss.setdefault("last_prompt", "")

    with st.sidebar:
        st.caption("문서를 올리고 Process를 누르면 문서 기반(RAG) 모드로 답해요.")
        uploaded = st.file_uploader("문서 업로드", type=["pdf", "docx", "json", "txt"], accept_multiple_files=True)
        if st.button("Process"):
            if not uploaded:
                st.warning("먼저 파일을 업로드해 주세요.")
            else:
                try:
                    docs = get_text(uploaded)
                    total = sum(len(d.page_content) for d in docs)
                    if total <= SMALL_DOC_CHARS:
                        ss.doc_text = "\n\n".join(d.page_content for d in docs)
                        ss.retriever = None
                    else:
                        ss.retriever = build_retriever(docs)
                        ss.doc_text = None
                    st.success("문서 처리가 완료되었습니다.")
                except ImportError:
                    st.error("큰 문서 검색에는 faiss-cpu, sentence-transformers가 필요해요. (로컬 환경에서 설치)")
                except Exception as e:
                    st.error(f"문서 처리 중 오류: {e}")

        if ss.doc_text or ss.retriever is not None:
            st.success("RAG 모드: 문서 기반 답변 중")
            if st.button("문서 끄기 (학습 모델만 사용)"):
                ss.doc_text = None
                ss.retriever = None
                st.rerun()
        else:
            st.info("기본 모드: 학습된 모델만 사용")
        if st.button("대화 초기화"):
            ss.messages = []
            st.rerun()
        with st.expander("마지막 프롬프트 보기"):
            st.code(ss.last_prompt or "(아직 없음)")

    st.chat_message("assistant").write(GREETING)
    for m in ss.messages:
        st.chat_message(m.role).write(m.content)

    if user_input := st.chat_input("메시지를 입력해 주세요"):
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            box = st.empty()
            try:
                kind = identity_kind(user_input)
                if kind and IDENTITY_MODE == "fixed":
                    ss.last_prompt = "(모델 정보 질문: 정해 둔 문장으로 바로 답변)"
                    final = FIXED[kind]
                    box.markdown(final)
                else:
                    ctx = MODEL_INFO_DOC if kind else get_context(user_input)
                    prompt = UNIFIED.format(context=ctx, history=get_history(), question=user_input)
                    ss.last_prompt = prompt
                    chain = RemoteRunnable(LLM_URL) | StrOutputParser()
                    parts = []
                    for chunk in chain.stream(prompt):
                        parts.append(chunk)
                        box.markdown(clean_answer("".join(parts)))
                    final = clean_answer("".join(parts)) or "(응답이 비어 있어요)"
                ss.messages.append(ChatMessage(role="user", content=user_input))
                ss.messages.append(ChatMessage(role="assistant", content=final))
            except Exception as e:
                st.error(f"답변 생성 중 오류: {e}\n\n서버(server_by.py)와 주소({LLM_URL})를 확인해 주세요.")


if __name__ == "__main__":
    main()
