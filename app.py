import streamlit as st
import os
import json
import datetime
from sentiment_engine import SentimentAnalyzer
from bot_logic import SupportBot

# --- CONFIGURATION ---
st.set_page_config(
    page_title="SentiBOT", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DARK THEME & MINIMAL CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    
    /* Sidebar Background */
    [data-testid="stSidebar"] { background-color: #262730; }
    
    /* Chat Bubbles */
    .stChatMessage {
        background-color: transparent;
        border: 1px solid #444;
        border-radius: 10px;
    }
    
    /* Minimal Mood Badge CSS */
    .mood-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 14px;
        font-weight: bold;
        color: white;
        margin-left: 10px;
    }
    
    /* History Dropdown Styling */
    div[data-baseweb="select"] > div {
        background-color: #363840;
        color: white;
        border-color: #555;
    }
</style>
""", unsafe_allow_html=True)

# --- SETUP ---
SESSION_DIR = "sessions"
if not os.path.exists(SESSION_DIR): os.makedirs(SESSION_DIR)

@st.cache_resource
def load_engines():
    return SentimentAnalyzer(), SupportBot()

analyzer, bot = load_engines()

# Initialize State
if "messages" not in st.session_state: st.session_state.messages = []
if "scores" not in st.session_state: st.session_state.scores = []

# --- HELPER FUNCTIONS ---
def save_chat():
    if not st.session_state.messages: return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"{SESSION_DIR}/chat_{timestamp}.json"
    with open(path, "w") as f:
        json.dump({"messages": st.session_state.messages, "scores": st.session_state.scores}, f)

def load_chat(filename):
    with open(os.path.join(SESSION_DIR, filename), "r") as f:
        data = json.load(f)
        st.session_state.messages = data["messages"]
        st.session_state.scores = data.get("scores", []) 

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 SentiBOT")
    
    # 1. NEW CHAT BUTTON
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        save_chat() # Auto-save current before clearing
        st.session_state.messages = []
        st.session_state.scores = []
        st.rerun()

    st.divider()

    # 2. MINIMAL MOOD INDICATOR
    st.caption("CURRENT VIBE")
    
    if st.session_state.scores:
        last_score = st.session_state.scores[-1]
        
        # Minimal Color Logic
        if last_score >= 0.05:
            color = "#00C853" # Bright Green
            label = "Positive"
        elif last_score <= -0.05:
            color = "#D50000" # Bright Red
            label = "Negative"
        else:
            color = "#757575" # Gray
            label = "Neutral"
            
        # Render Minimal Badge
        st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <span style="font-size: 20px;">{label == 'Positive' and '😊' or label == 'Negative' and '😡' or '😐'}</span>
            <span class="mood-badge" style="background-color: {color};">{label} ({last_score:.2f})</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:gray; font-style:italic; font-size:13px;'>Waiting for input...</span>", unsafe_allow_html=True)

    st.divider()
    
    # 3. RESTORED HISTORY (Old Chats)
    st.caption("HISTORY")
    files = sorted(os.listdir(SESSION_DIR), reverse=True)
    selected_file = st.selectbox("Select Session", files, index=None, label_visibility="collapsed", placeholder="Load past chat...")
    
    if selected_file:
        if st.button("📂 Load", use_container_width=True):
            load_chat(selected_file)
            st.rerun()

    st.divider()

    # 4. END SESSION & REPORT
    if st.button("🛑 End & Analyze"):
        report = analyzer.generate_session_report(st.session_state.scores)
        
        st.success("Session Analyzed!")
        
        st.markdown(f"""
        **Overall Verdict:** {report['verdict']}
        
        **Emotional Trend:** {report['trend']}
        
        **Avg Confidence:** {report['final_score']:.2f}
        """)
        
        # VISUAL GRAPH (Only appears here now)
        st.caption("Emotional Journey Graph")
        st.line_chart(st.session_state.scores, height=150)
        
        save_chat()
        st.caption("Session Saved.")

# --- MAIN CHAT ---
st.subheader("Real-time Sentiment Analysis Chat")

# Render History
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Input Loop
if prompt := st.chat_input("Type here..."):
    
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Analyze Sentiment
    sentiment = analyzer.analyze(prompt)
    st.session_state.scores.append(sentiment['score'])

    # 3. Streaming Bot Response
    with st.chat_message("assistant", avatar="🧠"):
        # Stream response from Ollama
        full_response = st.write_stream(
            bot.get_response_stream(prompt, sentiment['label'])
        )
    
    # Save full response to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Force rerun to update the Sidebar Mood immediately
    st.rerun()
