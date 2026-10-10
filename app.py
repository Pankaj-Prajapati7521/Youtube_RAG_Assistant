import os 
import re
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from dotenv import load_dotenv

load_dotenv(".env", override = True)
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube RAG Assistant")
st.write(
    "Ask questions about a YouTube video using "
    "Retrieval-Augmented Generation (RAG)."
)

#building llm
@st.cache_resource
def load_llm():
    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
    )
    return model

model = load_llm()

#building embeddings
@st.cache_resource
def load_embedding():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings =load_embedding()

#URL to video ID
def extract_video_id(video_input): 
    video_input = video_input.strip() 
    # Case 1: Already a video ID 
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_input): 
        return video_input
    # Case 2: Standard YouTube URL 
    patterns = [ 
        r"(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})", 
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})", 
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})", 
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})", 
    ]
    for pattern in patterns: 
        match = re.search(pattern, video_input) 
        if match: 
            return match.group(1) 
    return None


#building transcript(context)
def get_transcript(video_id):
    #video_id = st.text_input("Enter the video id")

    yt_api = YouTubeTranscriptApi()
    transcript_list  = yt_api.fetch(video_id, languages = ['en'])
    transcript = " ".join(chunk.text for chunk in transcript_list)

    return transcript

#building vector store
@st.cache_resource
def create_vector_store(transcript):

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store, len(chunks)

#video input
video_url = st.text_input(
    "🎬 Enter YouTube  Video ID",
    placeholder="https://www.youtube.com/watch?v=..."
)

#button for stat processing
if st.button("⚙️ Process Video"):
    if not video_url.strip():
        st.warning("Please enter a YouTube URL or Video ID.")
    else:
        video_id = extract_video_id(video_url)
        if not video_id: 
            st.error( "Invalid YouTube URL or Video ID." ) 
        else:
            try:
                with st.spinner("Fetching Transcript..."):
                    transcript = get_transcript(video_id)
                    with st.spinner("Creating embeddings and vector database..."):
                        vector_store, num_chunks = (create_vector_store(transcript))
                    st.session_state.vector_store = vector_store
                    st.session_state.video_id = video_id
                    st.session_state.transcript = transcript
                    st.session_state.num_chunks = num_chunks
                    st.session_state.video_processed = True
                    
                    st.success(
                        f"Video processed successfully! "
                        f"Created {num_chunks} chunks."
                    )

            except TranscriptsDisabled:
                st.error("Captions are disabled for this video.")

            except Exception as e: 
                st.error( "Unable to process this video." ) 
                st.info( "Make sure the video exists and has an "
                    "English transcript available." 
                ) 
            
                with st.expander("Technical details"): 
                    st.exception(e)

#taking question as input
question = st.text_input(
    "💬 Ask a question or query about the video",
    placeholder="What is this video about?"
)

#button for query
if st.button("Ask Question"):
    if not question.strip():
        st.warning("Please enter the question or query")
    elif "vector_store" not in st.session_state:
        st.warning("Please process a YouTube video first.")
    else:
        vector_store = (st.session_state.vector_store)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        def format_docs(retriever_docs):
            context_text = "\n\n".join(doc.page_content for doc in retriever_docs)
            return context_text

        parallel_chain = RunnableParallel({
            'context' : retriever | RunnableLambda(format_docs),
            'question' : RunnablePassthrough()
        })

        prompt = PromptTemplate(
        template = """You are a helpful assistant answering questions about a YouTube video.
            Use ONLY the provided transcript context to answer.
            Rules:
            1. Do not use outside knowledge.
            2. If the answer cannot be found in the context, say:
            "I don't know based on this video."
            3. Give a concise and clear answer.

            Context:
            {context}

            Question:
        {question}
        """,
        input_variables = ['context', 'question'] 
        )

        parser = StrOutputParser()
        
        final_chain = parallel_chain | prompt | model | parser

        with st.spinner("Generating answer..."):
            answer = final_chain.invoke(question)

        st.subheader("🤖 Answer")
        st.write(answer)

#retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {'k' : 4})
