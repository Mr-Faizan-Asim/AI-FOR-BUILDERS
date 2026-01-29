from flask import Flask, request, jsonify, render_template, session
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = "your_super_secret_key_here" 

# 1. Setup
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)

with open("regulations/fire_safety.md") as f:
    regulation_text = f.read()
with open("prompts/system_prompt.txt") as f:
    system_prompt = f.read()

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash", 
    system_instruction=f"{system_prompt}\n\nReference Regulations:\n{regulation_text}"
)

@app.route("/")
def home():
    session['history'] = []
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"reply": "No message provided."}), 400

    # Retrieve history and format it for Gemini
    history = session.get('history', [])

    try:
        # Start chat with the history (which is now just a list of dicts)
        chat = model.start_chat(history=history)
        
        # Send message
        response = chat.send_message(user_input)
        
        # --- CRITICAL FIX: Convert Protobuf objects to plain JSON-serializable dicts ---
        serializable_history = []
        for content in chat.history:
            serializable_history.append({
                "role": content.role,
                "parts": [part.text for part in content.parts]
            })
        
        # Save the "cleaned" history back to session
        session['history'] = serializable_history
        
        return jsonify({"reply": response.text})
    
    except Exception as e:
        print(f"Error details: {e}") # This helps you see the error in your terminal
        return jsonify({"reply": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)