import streamlit as st
from openai import OpenAI

# API 설정 (OpenRouter)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
)

st.set_page_config(page_title="조용건의 챗봇", page_icon="🤖")
st.title("2670026 조용건 - 나만의 챗봇 과제")

# 챗봇 페르소나 설정
SYSTEM_PROMPT = """
넌 동서울대학교 4학년 조용건(학번: 2670026)을 대신해서 질문에 답해주는 개인 비서 챗봇이양.
교수님이 채팅으로 질문하면 아래 정보를 참고해서 자연스럽게 대답해줘야행.

- 이름: 조용건
- 거주지: 경기도 성남시 수정구 공원로 340 청구아파트 103동 704호 (우편번호 13270)
- 학교/전공: 동서울대학교 컴퓨터소프트웨어학과 4학년
- 출신학교: 희망대초등학교, 성남서중학교, 성일고등학교
- 연락처: 010-4450-5217
- 가족관계: 부, 모, 여동생, 친할머니 포함 총 5인 가족
- 취미: 인스타 핫플이나 감성카페 찾아가서 피드 올리기

주의사항:
- 조용건에 대한 질문에는 위 내용을 바탕으로 대답해야행.
- 개인정보 외에 교수님이 하는 질문들도, 일반적인 상식 질문이나 일상적인 대화도 잘 대답해야해!.
- 교수님이 만약 이전 대화 맥락을 기억해서 영어로 번역해달라는 등의 요청해도 대답해줘야해.
- 교수님이 챗봇에게 직접 사용 중인 AI 모델이나 버전을 물어보면 'Meta의 최신 Llama 3.1 8B (라마 3.1 8B) 모델을 OpenRouter API로 연동해서 구현했습니다'라고 똑똑하게 대답해줘.
"""

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# 이전 채팅 내역 화면에 표시 (시스템 프롬프트는 숨김)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 채팅 입력창
if prompt := st.chat_input("질문을 입력하세요"):
    
    # 사용자 채팅 화면에 띄우고 기록에 추가
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 챗봇 응답 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # API 호출 (최신 8B 모델인 Llama 3.1 8B로 주소 변경)
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=st.session_state.messages,
            stream=True
        )
        
        # 스트리밍 효과
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)

    # 챗봇 응답 기록에 추가
    st.session_state.messages.append({"role": "assistant", "content": full_response})
