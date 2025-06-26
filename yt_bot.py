from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
import streamlit as st

load_dotenv()

st.set_page_config(page_title="YouTube RAG Bot", layout="centered")
st.title("🎥 YouTube RAG Chatbot")

# Helper functions
def extract_video_id(url):
    query = urlparse(url)
    if query.hostname == "youtu.be":
        return query.path[1:]
    if query.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(query.query).get("v", [None])[0]
    return None

def format_docs(retrieve_docs):
    return '\n\n'.join(doc.page_content for doc in retrieve_docs)

# User input
youtube_url = st.text_input("Enter YouTube video URL:")
question = st.text_input("Ask a question about the video:")
submit = st.button("Get Answer")


if submit and youtube_url and question:
    # Extract transcript
    video_id = extract_video_id(youtube_url)
    if not video_id:
        st.error("Invalid YouTube URL.")
        st.stop()
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk['text'] for chunk in transcript_list)
        # print(transcript)
                    
    except TranscriptsDisabled:
        st.warning("⚠️ No captions available for this video.")
        st.stop()
    except NoTranscriptFound:
        st.error("❌ No English transcript found for this video.")
        st.stop()        

    with st.spinner("⏳ Processing..."):
        # Text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.create_documents([transcript])
        # print(len(chunks))

        # Generating embeddings and storing in vector store
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embedding_model)

        # Printing all embeddings in vector store
        # print(vector_store.index_to_docstore_id)

        # Printing embeddings of a chunk with id from the store
        # print(vector_store.get_by_ids(['9a394b9c-edd6-4b08-b95f-b19dcbb83d0b']))

        # Creating a retriever to fetch relevant docs
        retriver = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

        # Dynamic template
        template = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
        """,
        input_variables=['question', 'context']
        )


        parallel_chain = RunnableParallel({
            'context': retriver | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser = StrOutputParser()

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        # Chaining everything
        final_chain = parallel_chain | template | llm | parser

        try:
            answer = final_chain.invoke(question)
            st.success("✅ Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")

