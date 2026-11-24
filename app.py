from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq

import os

# Load environment variables


app = Flask(__name__)

# Initialize Groq model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key="apikey ",
    temperature=0.7,
    max_retries=2
)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():

    try:
        data = request.get_json()

        if not data or not data.get("message"):
            return jsonify({"error": "Message is required"}), 400

        user_message = data["message"]

        # Generate response using LangChain
        response = llm.invoke(user_message)

        return jsonify({
            "response": response.content
        })

    except Exception as e:
        print("Error:", str(e))

        return jsonify({
            "error": "Something went wrong. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
