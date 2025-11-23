import ollama

class SupportBot:
    def __init__(self):
        self.model_name = "llama3" # Ensure you have this pulled: 'ollama pull llama3'

    def get_response_stream(self, user_text, sentiment_label):
        """
        Generates a streaming response from Ollama.
        """
        # 1. Adaptive System Prompts
        if sentiment_label == "Negative":
            sys_prompt = "You are SentiBOT, an empathetic AI. The user is upset. Be apologetic, short, and helpful."
        elif sentiment_label == "Positive":
            sys_prompt = "You are SentiBOT, a cheerful AI. Match the user's energy! Keep it concise."
        else:
            sys_prompt = "You are SentiBOT, a professional AI assistant. Be direct and helpful."

        # 2. Stream from Ollama
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': user_text},
                ],
                stream=True,  # <--- Enables streaming
            )
            
            # Yield each chunk of text as it arrives
            for chunk in stream:
                yield chunk['message']['content']
                
        except Exception as e:
            yield f"(Error: Ensure Ollama is running). Details: {e}"