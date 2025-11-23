# **🧠 SentiBOT: Sentiment-Aware AI Support Agent**

*A modular, adaptive AI agent that detects user emotion in real-time and adjusts its personality accordingly.*

## **🚀 Architecture Evolution**

This project evolved through two distinct phases to ensure both speed and accuracy.

### **Phase 1: The MVP (Initial Prototype)**

* **Interface:** Command Line Interface (CLI).  
* **Brain:** NLTK VADER (Rule-based sentiment analysis).  
* **Logic:** Hardcoded if/else responses.  
* **Limitation:** VADER failed to detect sarcasm or slang context (e.g., input *"I am about to party fuck yeaaa"* was detected as **Negative** due to the profanity).

### **Phase 2: The Enterprise Solution (Current Version)**

* **Interface:** **Streamlit Web App** (Dark Mode, Gemini-style UI).  
* **Brain:** **RoBERTa Transformers** (cardiffnlp/twitter-roberta-base-sentiment-latest).  
  * *Improvement:* Now understands context. *"I am about to party fuck yeaaa"* is correctly identified as **Positive (0.98)**.  
* **Logic:** **Ollama (Llama 3\)**.  
  * *Innovation:* **Adaptive System Prompts**. The system injects a different "personality" into the LLM based on the user's mood (e.g., if User is Angry \-\> Bot becomes Apologetic & Concise).

## **✨ Key Features**

1. **Real-Time Sentiment Streaming:**  
   * The bot doesn't just wait; it types out responses in real-time (Streaming API), providing a fluid user experience similar to ChatGPT.  
2. **Adaptive AI Personality:**  
   * **User is Happy:** Bot becomes cheerful and energetic.  
   * **User is Angry:** Bot becomes empathetic, apologetic, and solution-oriented.  
   * **User is Neutral:** Bot is professional and direct.  
3. **Persistent Session Memory:**  
   * Conversations are automatically saved as JSON files in the sessions/ directory.  
   * Users can load previous chats from the sidebar history dropdown.  
4. **Visual Mood Indicator:**  
   * A minimal sidebar badge updates instantly to reflect the current "Vibe" of the chat (Green/Red/Gray).

## **🛠️ Installation & Setup**

### **Prerequisites**

1. **Python 3.10+**  
2. **Ollama** (for the LLM backend). Download from [ollama.com](https://ollama.com).

### **Step 1: Install Dependencies**

pip install \-r requirements.txt

*Note: This installs torch, transformers, streamlit, and scipy.*

### **Step 2: Setup Local LLM**

Open your terminal/command prompt and run:  
ollama pull llama3

*This downloads the Llama 3 model (approx 4GB) to your machine so the bot works offline.*

### **Step 3: Run the Application**

streamlit run app.py

The app will open automatically in your browser at http://localhost:8501.

## **🧩 Technical Details**

### **Flexible LLM Backend**

While this project currently uses **Ollama (Llama 3\)**, the architecture is designed to be **LLM-Agnostic**.

* **Why Ollama?** It was chosen for this demonstration to ensure **offline capabilities**, **zero latency**, and **privacy** (no data leaves the machine).  
* **Scalability:** In a production environment, bot\_logic.py can be easily swapped to use **OpenAI API (GPT-4)**, **Google Gemini API**, or **Anthropic Claude** simply by changing the endpoint. The core logic (Adaptive System Prompts) remains the same regardless of the model used.

### **The Sentiment Pipeline (sentiment\_engine.py)**

We utilize the **RoBERTa** model trained on \~124 million tweets.

* **Input:** "This service is sick\!"  
* **Tokenization:** Splits text into sub-words.  
* **Inference:** Runs through the Transformer network.  
* **Output:** \[0.01, 0.04, 0.95\] (Probabilities for Neg, Neu, Pos).  
* **Verdict:** Positive.

*Windows Fix applied:* A patch for torch.classes.\_\_path\_\_ is included to ensure stability on Windows environments.

### **The Orchestrator (bot\_logic.py)**

Acts as the bridge between the Math (Sentiment) and the Creative (LLM).  
if sentiment \== "Negative":  
    system\_prompt \= "You are empathetic and apologetic..."  
else:  
    system\_prompt \= "You are cheerful and helpful..."

## **🧪 Testing Results**

| Input Phrase | VADER Score (Old) | RoBERTa Score (New) | Result |
| :---- | :---- | :---- | :---- |
| "This is fine." | Neutral | Neutral | ✅ Accurate |
| "I hate waiting." | Negative | Negative | ✅ Accurate |
| "Your service is sick\!" | Negative (Error) | **Positive** | 🚀 **Improved** |
| "Yeah, great job breaking it." | Positive (Error) | **Negative** | 🚀 **Sarcasm Detected** |

## **📂 File Structure**

LiaPlus\_Chatbot/  
├── app.py                  \# Main Streamlit Application (UI)  
├── sentiment\_engine.py     \# RoBERTa Transformer Logic  
├── bot\_logic.py            \# Ollama Generation Logic  
├── requirements.txt        \# Project Dependencies  
├── README.md               \# Documentation  
└── sessions/               \# JSON storage for chat history

**Tech Stack:** Python, Streamlit, PyTorch, Hugging Face, Ollama.