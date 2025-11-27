
import os
from openai import OpenAI

try:
    print("Attempting init with None...")
    client = OpenAI(api_key=None)
    print("Init success")
except Exception as e:
    print(f"Init failed: {e}")
