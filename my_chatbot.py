import streamlit as st
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
)


st.set_page_config(page_title="조용건님의 맞춤형 챗봇", page_icon="🤖")
st.title("2670026 - 조용건 나만의 챗봇 과제")


SYSTEM_PROMPT = """
너는 '조용건'을 소개하는 맞춤형 AI 챗봇이야. 
아래의 정보를 바탕으로 사용자의 질문에 친절하고 자연스럽게 답변해줘.

[개인정보]
- 이름: 조용건
- 고향: 경기도 성남
- 대학교: 동서울대학교
- 초등학교: 희망대초등학교
- 중학교: 성남서중학교
- 고등학교: 성일고등학교
- 학년: 4학년
- 학번: 2670026
- 전화번호: 010-4450-5217
- 주소: 경기도 성남시 수정구 공원로 340 청구아파트 103동 704호
- 우편번호: 13270
- 전공: 컴퓨터소프트웨어학과
- 취미: 인스타 핫플레이스 찾아가기, 피드올리기, 감성카페 찾아가기
- 가족: 부,모,여동생,친할머니,나 총 5인가족 

[규칙]
1. 위 [개인정보]에 있는 내용만 대답해.
2. 정보에 없는 내용을 물어보면 "해당 정보는 조용건님과 다른 정보입니다"라고 대답해.
"""


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if prompt := st.chat_input("질문을 입력하세요"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=st.session_state.messages,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
