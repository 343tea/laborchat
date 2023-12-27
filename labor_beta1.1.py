from openai import OpenAI
import pinecone
import streamlit as st
from firebase_admin import _apps, credentials, initialize_app, firestore
import uuid


def connect_to_openai():
    return OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

def connect_to_pinecone():
    pinecone.init(
        api_key=st.secrets['PINECONE_API_KEY'],
        environment=st.secrets['PINECONE_ENVIRONMENT'],
    )
    index_name = 'labordb'
    return pinecone.Index(index_name)

def connect_to_firebase():
    if not _apps:
        firebase_admin = st.secrets['FIREBASE_ADMIN_KEY']
        cred = credentials.Certificate(dict(firebase_admin))
        app = initialize_app(cred)
    db = firestore.client()
    info = db.collection('info').document('info2').get().to_dict()
    return db, info

def query_to_embedding(user_input):
    embed = client.embeddings.create(
        input=user_input,
        model='text-embedding-ada-002',
    )
    qvector = embed.data[0].embedding
    return qvector

def vector_similarity(qvector):
    nearest = index.query(
        namespace='test1-a',
        vector=qvector,
        top_k=5,
        include_values=False,
        include_metadata=True,
    )
    return nearest

def save_to_firebase(db, session_id, question, answer):
    timestamp = firestore.SERVER_TIMESTAMP
    history = {'timestamp': timestamp, 'session_id': session_id, 'qustion': question, 'answer': answer}
    db.collection('chat_history').add(history)


st.title('노무 상담 챗봇')
st.subheader('version beta1.10')

client = connect_to_openai()
index = connect_to_pinecone()
db, info = connect_to_firebase()

# 초기화할 세션 상태 변수들과 그 기본값 정의
default_values = {
    'session_id': str(uuid.uuid4()),
    'openai_model': 'gpt-3.5-turbo-16k',
    'question_count': 0,
    'messages': [{'role': 'system', 'content': info['role']}]
}

# session_state 설정
for key, value in default_values.items():
    st.session_state.setdefault(key, value)

# 안내 문구 표시
st.info(info['guide'])

# 기존 메시지 출력
for message in st.session_state.messages:
    if message['role'] != 'system':
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# 사용자 입력 처리
if prompt := st.chat_input('질문을 입력하세요.'):
    # 질문 횟수 카운트
    st.session_state.question_count += 1

    if st.session_state.question_count < 1000:
        qvector = query_to_embedding(prompt)
        nearest = vector_similarity(qvector)

        chunk_list = []
        # nearest에서 chunk를 뽑아서 출력
        for i in range(len(nearest.matches)):
            chunk = nearest.matches[i].metadata['chunk']
            chunk_list.append(chunk)

            # 첫번째 청크의 제목과 링크를 출력
            if i == 0:
                title = chunk.split('|')[0]
                link = nearest.matches[i].metadata['link']
                best = f'\n\n▲ 관련 내용 : [{title}](https://{link})'

        st.session_state.messages.append({'role': 'system', 'content': f'관련 청크: {chunk_list}'})
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # 사용자 질문을 출력
        with st.chat_message('user'):
            st.markdown(prompt)

        # 답변을 생성해 출력
        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            full_response = ''

            # 대화 내역 관리: 가장 최근의 N개의 메시지만 유지
            recent_messages = st.session_state.messages[-2:]  # 예시: 질문-답변 최근 2쌍만 유지

            # OpenAI API 호출해 답변 생성
            for response in client.chat.completions.create(
                model=st.session_state['openai_model'],
                messages=[
                    {'role': m['role'], 'content': m['content']}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                # 답변이 생성되는 대로 갱신하며 출력
                full_response += (response.choices[0].delta.content or '')
                message_placeholder.markdown(full_response + '▌')

            # 전체 답변을 출력하면서 끝에 관련 내용(링크) 추가
            completed_response = full_response + best
            message_placeholder.markdown(completed_response)

        # Firebase에 질문과 답변 저장
        save_to_firebase(db, st.session_state['session_id'], prompt, completed_response)

        # 답변을 session_state의 message 목록에 추가
        st.session_state.messages.append({'role': 'assistant', 'content': completed_response})

    else:
        # 질문 한도를 초과한 경우
        response = info['limit']
        with st.chat_message('assistant'):
            st.markdown(response)

