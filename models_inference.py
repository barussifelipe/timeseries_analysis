from google import genai 
import os 
from dotenv import load_dotenv

load_dotenv() 

gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

client = genai.Client(api_key= gemini_key)