# YouTube RAG Assistant 🎥

An AI-powered application that allows users to ask questions about YouTube videos and get answers based on their transcripts.

## Features
- Enter a YouTube video URL or ID.
- Fetch video transcripts.
- Split transcripts into text chunks.
- Use Hugging Face embeddings and FAISS to find relevant information.
- Generate answers using Groq LLM.
- Simple user interface built with Streamlit.

## Tech Stack
- Python
- Streamlit
- LangChain
- Hugging Face
- FAISS
- Groq API
- YouTube Transcript API

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Pankaj-Prajapati7521/youtube_rag_assistant.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your `GROQ_API_KEY` in your environment or Streamlit Secrets.

4. Run the application:

   ```bash
   streamlit run app.py
   ```

## Live Demo

[Try YouTube RAG Assistant](https://youtuberagassistant-uzpouzerxahejfzqrri4nq.streamlit.app/)

## Author

**Pankaj Prajapati**

[GitHub Profile](https://github.com/Pankaj-Prajapati7521)
