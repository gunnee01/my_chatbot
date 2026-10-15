import streamlit as st
from openai import OpenAI

# OpenRouter API 클라이언트 설정
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
)

st.set_page_config(page_title="조용건의 챗봇", page_icon="🤖")
st.title("2670026 조용건 - 나만의 챗봇 과제")

# 챗봇 페르소나 및 지침 설정 (격식 있고 명확한 어조로 정제)
SYSTEM_PROMPT = """
너는 동서울대학교 컴퓨터소프트웨어학과 4학년 조용건(학번: 2670026)을 대신해서 질문에 답해주는 개인 비서 챗봇이야.
사용자(교수님)가 채팅으로 질문하면 아래 정보를 참고해서 친절하고 공손하게 대답해줘.

[조용건 개인정보]
- 이름: 조용건
- 거주지: 경기도 성남시 수정구 공원로 340 청구아파트 103동 704호 (우편번호 13270)
- 학교/전공: 동서울대학교 컴퓨터소프트웨어학과 4학년
- 출신학교: 희망대초등학교, 성남서중학교, 성일고등학교
- 연락처: 010-4450-5217
- 가족관계: 부, 모, 여동생, 친할머니 포함 총 5인 가족
- 취미: 인스타 핫플이나 감성카페 찾아가서 피드 올리기

[주의사항 및 답변 규칙]
1. 조용건에 대한 개인 신상 질문에는 위 [조용건 개인정보]를 바탕으로 정확하게 대답해줘.
2. 개인정보 외에 교수님이 하시는 일반적인 상식 질문(예: 대한민국 수도 등)이나 일상적인 대화에도 친절하게 대답해줘.
3. 교수님이 이전 대화 맥락을 기억해서 "위 대답을 영어로 번역해줘" 등의 요청을 하면 이전 대화 내용을 바탕으로 정확히 대답해줘.
4. 챗봇에게 직접 사용 중인 AI 모델이나 버전을 물어보면 "Meta의 최신 Llama 3.1 8B (라마 3.1 8B) 모델을 OpenRouter API로 연동해서 구현했습니다."라고 명확하게 대답해줘.
"""

# 대화 내용 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# 이전 채팅 내역 화면에 표시 (System 메시지 제외)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 채팅 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):

    # 1. 사용자 채팅 화면 출력 및 세션 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 챗봇 응답 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # OpenRouter API 호출 (최신 3.1 모델로 유지)
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=st.session_state.messages,
            stream=True
        )

        # 스트리밍 예외 방지 안전 처리 (노트북LM의 아이디어 + 문법 오류 수정)
        for chunk in response:
            # chunk.choices가 비어있지 않고, 첫 번째 요소[0]의 텍스트가 있을 때만 처리
            if chunk.choices and chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # 3. 챗봇 응답 세션 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})
