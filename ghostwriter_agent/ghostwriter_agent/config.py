# ghostwriter_agent/config.py
import os
from dotenv import load_dotenv
from google.adk.models.google_llm import Gemini

load_dotenv()  # loads GOOGLE_API_KEY from .env if present

model = Gemini(model="gemini-2.5-flash-lite")

