import os
import json
import uuid
import numpy as np
import faiss
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. Configuration & API Clients
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"

# File & Directory Paths
INDEX_FILE = "multi_pdf_faiss_index.bin"
DOCS_FILE = "multi_pdf_documents.json"
CHATS_FILE = "medicus_chats.json"
MODEL_FILE = "mri_Inceptionv3_f2x64.h5"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("templates", exist_ok=True)

# ==========================================
# 2. Load RAG FAISS Index & Diagnosis Model
# ==========================================
index = None
documents = []
if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)
else:
    print("WARNING: FAISS index or documents file missing. RAG will answer without vector context.")

diagnosis_model = None
CLASS_LABELS = ["glioma", "meningioma", "no tumor", "pituitary"]
if os.path.exists(MODEL_FILE):
    diagnosis_model = tf.keras.models.load_model(MODEL_FILE)
else:
    print(f"WARNING: Diagnosis model '{MODEL_FILE}' not found. /predict requires this file.")

# Initialize Session-Based Chat Database (No user/login storage)
if not os.path.exists(CHATS_FILE):
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)


# ==========================================
# 3. Helper Functions
# ==========================================
def load_chats():
    with open(CHATS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chats(data):
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def generate_chat_title(query):
    """Generates a concise 3-4 word title for a new RAG session."""
    try:
        res = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize this query into a short 3-4 word title. "
                        f"Respond ONLY with the title, no quotes or punctuation: {query}"
                    ),
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=15,
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return "Medical Consultation"


def predict_image(img_path):
    """Preprocesses an MRI image (299x299) and runs InceptionV3 classification."""
    if diagnosis_model is None:
        raise RuntimeError("Diagnosis model 'mri_Inceptionv3_f2x64.h5' is not loaded.")

    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = diagnosis_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_LABELS[predicted_class_index]
    return predicted_class, predictions


# ==========================================
# 4. Page Navigation Routes
# ==========================================
@app.route("/")
def home():
    """Renders the main Medicus AI console (AI Copilot + AI Assistant)."""
    return render_template("index_up.html")


@app.route("/diagnosis")
def diagnosis_page():
    """Renders the Diagnosis Models page from the sidebar navigation bar."""
    return render_template("index.html")


# ==========================================
# 5. RAG Framework Routes (Copilot & Assistant)
# ==========================================
@app.route("/api/sessions", methods=["GET", "POST"])
def get_sessions():
    """Returns all chat sessions without requiring authentication."""
    chats_db = load_chats()
    sessions = [
        {"id": sid, "title": data.get("title", "Medical Consultation")}
        for sid, data in chats_db.items()
    ]
    sessions.reverse()
    return jsonify({"sessions": sessions})


@app.route("/api/history", methods=["GET", "POST"])
def get_history():
    """Returns saved messages for a given session_id."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id") or request.args.get("session_id")

    chats_db = load_chats()
    messages = chats_db.get(session_id, {}).get("messages", [])
    return jsonify({"messages": messages})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Unified RAG endpoint for both:
      1. The floating AI Copilot widget (source='copilot')
      2. The full AI Assistant section (source='assistant')
    """
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        user_query = (payload.get("query") or "").strip()
        session_id = payload.get("session_id")
        source = payload.get("source", "assistant")
        attachment_name = None
    else:
        user_query = (request.form.get("query") or "").strip()
        session_id = request.form.get("session_id")
        source = request.form.get("source", "assistant")
        uploaded_file = request.files.get("file")
        attachment_name = None

        if uploaded_file and uploaded_file.filename:
            attachment_name = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(UPLOAD_FOLDER, attachment_name))

    if session_id in (None, "", "null", "undefined"):
        session_id = None

    if not user_query and not attachment_name:
        return jsonify({"error": "Missing query"}), 400

    if not user_query and attachment_name:
        user_query = f"Please review the attached file: {attachment_name}"

    chats_db = load_chats()
    is_new_session = False

    if not session_id or session_id not in chats_db:
        session_id = str(uuid.uuid4())
        title = generate_chat_title(user_query)
        chats_db[session_id] = {"title": title, "messages": []}
        is_new_session = True

    try:
        # Step A: Retrieve Relevant Knowledge Base Context via FAISS
        retrieved_contexts = []
        if index is not None and documents:
            emb_res = openai_client.embeddings.create(
                input=[user_query], model=EMBEDDING_MODEL
            )
            query_embedding = np.array([emb_res.data[0].embedding]).astype("float32")
            distances, indices = index.search(query_embedding, 2)

            for i, idx in enumerate(indices[0]):
                dist = distances[0][i]
                if idx != -1 and idx < len(documents):
                    if dist < 0.5:  # Threshold for relevance
                        retrieved_contexts.append(documents[idx])

        context_text = "\n\n".join(retrieved_contexts)

        # Step B: Load Recent Session Memory (Last 6 messages)
        past_conversation = chats_db[session_id]["messages"][-6:]

        # Step C: Build System & Conversation Prompt
        system_prompt = (
            "You are Medicus AI, an onboard aerospace medical copilot and clinical assistant. "
            "Respond clearly, calmly, and accurately using the retrieved medical knowledge base below, "
            "and offer relevant follow-up guidance.\n\n"
            f"CONTEXT FROM KNOWLEDGE BASE:\n{context_text}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(
            [{"role": m["role"], "content": m["content"]} for m in past_conversation]
        )

        prompt_query = (
            f"{user_query} [Attached file: {attachment_name}]"
            if attachment_name
            else user_query
        )
        messages.append({"role": "user", "content": prompt_query})

        # Step D: Generate Response using Groq
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_completion_tokens=500,
        )
        answer = chat_completion.choices[0].message.content

        # Step E: Persist Chat History
        chats_db[session_id]["messages"].append(
            {"role": "user", "content": prompt_query, "source": source}
        )
        chats_db[session_id]["messages"].append(
            {"role": "assistant", "content": answer, "source": source}
        )
        save_chats(chats_db)

        return jsonify(
            {
                "answer": answer,
                "session_id": session_id,
                "is_new_session": is_new_session,
                "title": chats_db[session_id]["title"],
            }
        )

    except Exception as e:
        print(f"RAG Chat Error: {e}")
        return jsonify({"error": "An internal error occurred while processing your query."}), 500


# ==========================================
# 6. Diagnosis Model Route (from app_d.py)
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():
    """Classifies brain MRI images and generates an AI diagnostic report."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        predicted_class, predictions = predict_image(file_path)

        ai_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are Medicus AI, an AI assistant for medical diagnosis.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Based on the MRI of the Brain of a person, the brain tumor type is '{predicted_class}'. "
                        "Give a short description about the tumor and also provide the reasons, symptoms, "
                        "necessary steps needs to be taken to cure and how to prevent it. "
                        "Don't give any introduction or warning about the response. "
                        "Note that no tumor means the person has no brain tumor issues."
                    ),
                },
            ],
        )

        diagnosis_report = ai_response.choices[0].message.content

        return jsonify(
            {
                "predicted_class": predicted_class,
                "prediction_probabilities": predictions.tolist(),
                "ai_model_response": diagnosis_report,
            }
        )
    except Exception as e:
        print(f"Diagnosis Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)