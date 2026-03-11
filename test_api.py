import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found in .env file")
    print("Please make sure your .env file contains: GEMINI_API_KEY=your_key_here")
    exit(1)

print(f"✅ API Key found: {API_KEY[:10]}...")
print("-" * 50)

# Configure the API
genai.configure(api_key=API_KEY)

# 1. List all available models
print("📋 AVAILABLE MODELS:")
print("-" * 50)
try:
    models = genai.list_models()
    for model in models:
        print(f"📌 Model: {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Supported Methods: {model.supported_generation_methods}")
        print()
except Exception as e:
    print(f"❌ Error listing models: {e}")

print("-" * 50)

# 2. Test specific models
test_models = [
    "models/gemini-1.5-pro",
    "models/gemini-1.5-flash",
    "models/gemini-pro",
    "models/gemini-pro-vision",
]

print("🧪 TESTING SPECIFIC MODELS:")
print("-" * 50)

for model_name in test_models:
    try:
        print(f"Testing: {model_name}")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'Hello' in one word")
        print(f"✅ SUCCESS: {response.text}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    print("-" * 30)

# 3. Test question generation with working model
print("\n🎯 TESTING QUESTION GENERATION:")
print("-" * 50)

try:
    # Use the flash model which usually works
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    
    prompt = """Generate 1 multiple choice question for Aptitude at easy level.
    
    Return ONLY a valid JSON array with this exact structure:
    [
        {
            "question": "What is 2+2?",
            "options": {
                "A": "3",
                "B": "4",
                "C": "5",
                "D": "6"
            },
            "correct_answer": "B",
            "explanation": "2+2 equals 4"
        }
    ]"""
    
    response = model.generate_content(prompt)
    
    if response and response.text:
        print("✅ Generated Response:")
        print(response.text)
    else:
        print("❌ No response generated")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("-" * 50)
print("\n✅ Test complete!")