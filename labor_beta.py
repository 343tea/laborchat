# version: beta2

from openai import OpenAI
import pinecone
import streamlit as st
from firebase_admin import _apps, credentials, initialize_app, firestore
import uuid
from datetime import datetime
import pytz

def connect_to_openai():
    return OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

def connect_to_pinecone():
    pinecone(
        api_key=st.secrets['PINECONE_API_KEY'],
    )
    index_name = st.secrets['PINECONE_INDEX_NAME']
    return pinecone.Index(index_name)

def connect_to_firebase():
    if not _apps:
        firebase_admin = st.secrets['FIREBASE_ADMIN_KEY']
        cred = credentials.Certificate(dict(firebase_admin))
        app = initialize_app(cred)
    db = firestore.client()
    info = db.collection('info').document('info').get().to_dict()
    return db, info

def save_to_firebase(db, session_id, question, answer, korea_timezone):
    current_time = datetime.now(korea_timezone)
    timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
    history = {'timestamp': timestamp, 'session_id': session_id, 'qustion': question, 'answer': answer}
    db.collection('chat_history').document(timestamp).set(history)

def query_to_embedding(user_input):
    embed = client.embeddings.create(
        input=user_input,
        model='text-embedding-ada-002',
    )
    qvector = embed.data[0].embedding
    return qvector

def vector_similarity(qvector, namespace, top_k):
    nearest = index.query(
        namespace=namespace,
        vector=qvector,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
    )
    return nearest

def extract_meta_list(nearest):
    return [match.metadata for match in nearest.matches]

def select_best(meta_list):
    # 청크와 링크 쌍을 저장할 리스트 초기화
    selected_titles = []

    # 중복 링크를 거르기 위한 집합
    processed_links = set()

    # 청크 리스트와 관련 내용 추출
    for meta in meta_list:
        if len(selected_titles) < 3:
            title = meta['title']
            link = meta['link']

            # 중복되지 않은 링크만 처리
            if link not in processed_links:
                processed_links.add(link)
                selected_titles.append((title, link))
        else:
            break

    # selected_titles를 사용해 best 문자열 생성
    best = ''
    for i, (title, link) in enumerate(selected_titles):
        best += f'\n\n▲ 관련 내용 {i + 1} : [{title}]({link})'
    return best


st.title(st.secrets['APP_NAME'])
st.subheader('v.beta2')

client = connect_to_openai()
index = connect_to_pinecone()
db, info = connect_to_firebase()
korea_timezone = pytz.timezone('Asia/Seoul')

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

    # 사용자 질문을 출력
    with st.chat_message('user'):
        st.markdown(prompt)

    if st.session_state.question_count < 1000:

        qvector = query_to_embedding(prompt)
        add_prompt = st.session_state.messages[-1]['content'] + '\n' + prompt
        add_qvector = query_to_embedding(add_prompt)

        # 혼합 검색
        nearest_q2q = vector_similarity(qvector, 'qna-q', 2)  # query to questions
        nearest_q2qa = vector_similarity(qvector, 'qna-qa', 1)  # query to questions and answers
        nearest_q2p = vector_similarity(qvector, 'test1-a', 3)  # query to pieces
        nearest_add2qa = vector_similarity(add_qvector, 'qna-qa', 1)  # add query to answers and questions

        meta_list = sum(
            [extract_meta_list(nearest) for nearest in [nearest_q2q, nearest_q2qa, nearest_q2p, nearest_add2qa]], [])

        # 관련 내용을 담은 청크 추출
        chunk_list = [meta['chunk'] for meta in meta_list]

        # 관련 내용을 담은 블로그 제목과 링크 추출
        best = select_best(meta_list)

        # 강제 교정 장치
        # chunk_list.append(info['note'])

        # 메시지 리스트에 추가할 메시지들
        messages_to_append = [
            {'role': 'system', 'content': info['role']},
            {'role': 'system', 'content': f'관련 내용: {chunk_list}'},
            {'role': 'user', 'content': prompt}
        ]

        for message in messages_to_append:
            st.session_state.messages.append(message)

        # 답변을 생성해 출력
        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            full_response = ''

            # 직전의 질문과 답변 포함해서 불러오기
            recent_messages = st.session_state.messages[-5:]

            # OpenAI API 호출해 답변 생성
            for response in client.chat.completions.create(
                    model=st.session_state['openai_model'],
                    messages=[
                        {'role': m['role'], 'content': m['content'].split('▲')[0]}
                        for m in recent_messages
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
        save_to_firebase(db, st.session_state['session_id'], prompt, completed_response, korea_timezone)

        # 답변을 session_state의 message 목록에 추가
        st.session_state.messages.append({'role': 'assistant', 'content': completed_response})

    else:
        # 질문 한도를 초과한 경우
        response = info['limit']
        with st.chat_message('assistant'):
            st.markdown(response)

