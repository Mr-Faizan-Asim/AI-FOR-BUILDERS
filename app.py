from flask import Flask, request, jsonify, render_template
import openai

app = Flask(__name__)

openai.api_key = "YOUR_API_KEY"

with open("regulations/fire_safety.md") as f:
    regulation_text = f.read()

with open("prompts/system_prompt.txt") as f:
    system_prompt = f.read()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json["message"]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": regulation_text},
            {"role": "user", "content": user_input}
        ]
    )

    return jsonify({"reply": response.choices[0].message.content})

if __name__ == "__main__":
    app.run(debug=True)
