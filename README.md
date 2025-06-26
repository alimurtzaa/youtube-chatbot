# 🎥 YouTube RAG Chatbot

This is a simple **Retrieval-Augmented Generation (RAG)** app built using **Streamlit**, **LangChain**, and **OpenAI**, which allows you to ask questions about any **YouTube video** that has **English captions**.

---

## ✨ Features

- ✅ Extracts English transcripts from YouTube videos
- ✅ Splits and embeds text using `text-embedding-3-small`
- ✅ Stores embeddings in a FAISS vector store
- ✅ Uses a retriever and GPT model (`gpt-4o-mini`) to answer questions
- ✅ Simple and clean Streamlit UI

---

## 🚀 How to Run

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/youtube-rag-chatbot.git
   cd youtube-rag-chatbot

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Create a .env file with your OpenAI API key:**
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   
4. **Run the app:**
   ```bash
   streamlit run yt_bot.py

## 📌 Notes
The app only works with videos that have English transcripts.
Videos with disabled or unavailable captions will return a friendly error.
