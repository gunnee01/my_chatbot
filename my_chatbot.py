import streamlit as st
from openai import OpenAI

# OpenRouter API 클라이언트 설정
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
)

st.set_page_config(page_title="조용건님의 맞춤형 챗봇", page_icon="🤖")

st.title("2670026 - 조용건 나만의 챗봇 과제")

# 프롬프트 수정: 개인정보를 우선 답변하되, 일반 질문 및 대화 맥락 기억 질문에도 답변하도록 설정
SYSTEM_PROMPT = """
너는 '조용건'을 소개하는 맞춤형 AI 챗봇이자 친절한 대화형 AI야.
아래의 정보를 바탕으로 사용자의 질문에 친절하고 간결하게 답변해줘.

[조용건 개인정보]
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
- 가족: 부, 모, 여동생, 친할머니, 나 총 5인가족

[답변 규칙]
1. '조용건'님이나 개인 신상에 관한 질문에는 위 [조용건 개인정보]를 바탕으로 정확하게 답변해.
2. 대한민국 수도, 상식 질문, 일반 대화, 그리고 "위의 대답을 영어로 바꿔줘" 같은 대화 맥락 요청에는 지식이 나타나도록 친절하고 간결하게 답변해줘.
3. 사용자가 사용 중인 AI 모델이나 버전을 물어보면 OpenRouter를 통한 AI 모델이라고 자연스럽게 대답해줘.
"""

# 세션 상태에 대화 히스토리 초기화 (System Prompt 포함)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# 화면에 이전 대화 출력 (system 메시지 제외)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):
    # 1. 사용자 질문 출력 및 히스토리에 추가
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. AI 응답 생성 (전체 대화 히스토리 전달)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # st.session_state.messages 전체를 넘겨주어 이전 대화 맥락을 기억하게 함
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=st.session_state.messages,
            stream=True
        )
        
        for chunk in response:
            # 수정된 부분: choices가 리스트 형태이므로 [0] 인덱스로 접근
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)

    # 3. AI 응답을 히스토리에 추가
    st.session_state.messages.append({"role": "assistant", "content": full_response})
