from dotenv import load_dotenv
import os

load_dotenv()

print("SID:", os.getenv("TWILIO_SID"))
print("Token:", os.getenv("TWILIO_AUTH_TOKEN"))
print("From:", os.getenv("TWILIO_FROM_NUMBER"))
print("To:", os.getenv("TO_NUMBER"))