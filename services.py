import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class AIService:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def chat_completion(self, messages, model="llama-3.1-8b-instant", stream=True):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
        )
