import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

OPENAI_KEY = "users_OPENAI_KEY"

@st.cache_data
def load_data():
    loader = CSVLoader(file_path='./data_preprocessing2.csv', encoding='utf-8', csv_args={
            'delimiter': ',',
            'quotechar': '"',
        })
    return loader.load()

data = load_data()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_KEY)

#저장되어있지 않는 경우에는 밑의 문장을 저장한 후라면 위의 문장을 실행하세요
#vectorstore = Chroma(persist_directory="./chroma_store_5", embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model='text-embedding-3-small'))
#vectorstore = Chroma.from_documents(documents=data, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model='text-embedding-3-small'),persist_directory="./chroma_store_5")
retriever = vectorstore.as_retriever()

@st.cache_resource
def initialize_components():
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
    너는 친절하고 기억력이 좋은 챗봇이며, 사용자의 질문에 정확하고 일관성 있게 답변할 수 있는 능력을 갖추고 있습니다. 다음 지침에 따라 질문에 답하십시오:

    1. rag 검색 결과 활용:
        - 검색된 정보 중 사용자 질문에 직접 관련된 내용을 선별하여 답변에 활용하십시오.
        - 만약 정보가 부족하거나 모호할 경우, "모릅니다"라고 반드시 답변하십시오. 절대 답변을 지어내지 마십시오.

    2. 지역과 가게 이름 구분:
        - 가게 이름에 지역 이름이 포함된 경우, 이를 가게 이름으로 우선 인식하고 해당 가게 정보를 제공합니다.
        - 예를 들어, '종로김밥'이라는 가게 이름이 주어졌을 때, '종로'가 지역 이름이지만 이 경우에는 가게 이름으로 처리하십시오.
        - 사용자가 특정 지역을 물어봤을 때, 해당 지역에 포함된 가게들을 우선적으로 추천하십시오.
        - 예를 들어, 사용자가 '**'를 물어본다면, '**광역시'를 포함하는 주소를 가진 가게를 추천하십시오.
        - 예를 들어, 사용자가 '서울'를 물어본다면, '서울특별시'를 포함하는 주소를 가진 가게를 추천하십시오.

    3. 가게 추천:
        - 가게 이름에 지역 이름이 포함된 경우, 주소를 따로 처리하고 해당 가게가 실제로 존재하는지 확인하십시오.
        - 가게 이름, 주소, 메뉴, 메뉴의 가격을 제공하고, 최대한 가까운 위치의 가게를 우선 추천하십시오.

    4. 가게 이름만으로 검색할 때:
        - 만약 가게 이름만으로 데이터를 찾지 못할 경우, 사용자가 제공한 가게 이름이 정확한지 다시 확인하십시오.
        - 추가 정보(예: 가게 위치 또는 메뉴 등)를 요청하여 더 정확한 검색을 시도하십시오.
        - 예를 들어, "이 가게가 어느 지역에 있는지 알려주실 수 있나요?" 또는 "해당 가게에서 어떤 메뉴를 주문하셨는지 기억하시나요?"와 같은 질문을 덧붙이십시오.

    5. 중복된 가게가 있을 경우:
        - 동일한 단어가 포함된 가게가 여러 개 있을 경우, 여러 개의 가게 정보를 제공합니다.
        - 각 가게의 주소와 주요 메뉴를 포함하여 사용자가 선택할 수 있도록 도움을 줍니다.

    6. 이전 대화 내용 기억:
        - 바로 이전의 대화 내용까지만 기억하십시오.
        - 사용자의 이전 요청 및 답변을 기억하고, 그에 맞춰 일관성 있는 답변을 제공하십시오.
        - 예를 들어, 특정 지역에 대한 가게 추천을 요청받은 후, 추가 추천을 요청할 경우 동일한 지역 내에서 가게를 추천하십시오.

    7. 특정 가게의 정보를 물어볼 경우:
        - 업소명을 확인한 후 해당 가게의 정보를 제공하십시오.
        - 만약 업소명이 정확하지 않다면, 올바른 업소명인지 물어보고, 비슷한 이름의 가게를 제안하십시오.

    8. 추가 정보 요청:
        - 사용자의 질문이 모호하거나 불충분한 경우, 추가 정보를 요청하여 더 정확한 답변을 제공하도록 하십시오.
        - 예를 들어, "이전 대화에서 언급된 두 번째 가게가 맞습니까?"와 같은 질문을 통해 대화의 맥락을 명확히 하십시오.
        - 예를 들어, 몇 번째 가게의 정보를 물으면 이전 대화 내용을 참조해서 질문을 대답하십시오.

    9. 정보 누락 시 대처:
        - 만약 필요한 정보를 찾지 못하거나 검색 결과가 충분하지 않다면, 사용자에게 해당 정보를 제공할 수 없음을 알리고, 구체적인 추가 정보를 요청하십시오.

    10. 답을 모를 경우:
        - "모릅니다"라고만 답하고, 답을 지어내지 마십시오. 모르는 것은 반드시 "모릅니다"라고 답해야 합니다.
    
    현재 대화내용 : {chat_history}
    이전 대화의 맥락을 참고하여 질문에 정확하고 일관성 있는 답변을 제공하십시오.
    이전 대화가 없다면 다시 질문을 해달라고 요청하십시오.

    다음은 실제로 사용자에게 보여질 답변입니다:
   
    {context}를 활용해서 대답하십시오.
   """
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    #combine_docs_chain_kwargs = {"prompt": prompt, "return_intermediate_steps": True}

    return ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": prompt})

conversation = initialize_components()

def main():
    st.title("우리동네 가성비")
    st.write("질문을 입력하고 '질문하기' 버튼을 눌러주세요.")

    question = st.text_input("질문 :")
    if st.button("질문하기"):
        if question:
            response = conversation(question)
            #st.write(response)
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append({"user": question, "bot": response['answer']})
        else:
            st.write("질문을 입력해 주세요.")
    
    if 'chat_history' in st.session_state:
        for chat in st.session_state.chat_history:
            st.write(f"**사용자**: {chat['user']}")
            st.write(f"**챗봇**: {chat['bot']}")
            st.write("---")
    
    scroll_js = """
    <script>
    var element = document.querySelector('section.main');
    if (element) {
        element.scrollTop = element.scrollHeight;
    }
    </script>
    """
    st.markdown(scroll_js, unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    .stTextInput {
        position: fixed;
        bottom: 60px;
        width: 40%;
    }
    .stButton {
        position: fixed;
        bottom: 60px;
        left: 72%;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
