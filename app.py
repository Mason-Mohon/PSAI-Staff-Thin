from flask import Flask, render_template, request, jsonify, send_file, redirect, session, g
import os
import io
import json
import time
import uuid
import logging
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests as http_requests
from dotenv import load_dotenv
import qdrant_client
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import jwt
from jwt import PyJWKClient
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- App ----------
app = Flask(__name__)
load_dotenv()

# ---------- Configuration ----------
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")

# Cognito
COGNITO_REGION = os.environ.get("COGNITO_REGION", "us-east-2")
COGNITO_USER_POOL_ID = os.environ.get("COGNITO_USER_POOL_ID", "")
COGNITO_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID", "")
COGNITO_DOMAIN = os.environ.get("COGNITO_DOMAIN", "")
REDIRECT_URI = os.environ.get("COGNITO_REDIRECT_URI", "https://ai.phyllisschlafly.com/callback")

ISSUER = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}" if COGNITO_USER_POOL_ID else ""
DOMAIN = f"https://{COGNITO_DOMAIN}.auth.{COGNITO_REGION}.amazoncognito.com" if COGNITO_DOMAIN else ""

# Feature flags
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "0") == "1"

# ---------- Clients ----------
qdrant = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------- JWT / Cognito helpers ----------
_jwks_client = None

def _get_jwks_client():
    global _jwks_client
    if _jwks_client is None and ISSUER:
        _jwks_client = PyJWKClient(f"{ISSUER}/.well-known/jwks.json")
    return _jwks_client

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify a Cognito JWT id_token. Returns decoded payload or None."""
    try:
        if not ISSUER or not COGNITO_CLIENT_ID or not token:
            return None
        jwks = _get_jwks_client()
        if not jwks:
            return None
        signing_key = jwks.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=COGNITO_CLIENT_ID,
            issuer=ISSUER,
            options={"verify_signature": True, "verify_exp": True},
        )
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid token: %s", e)
        return None
    except Exception as e:
        logger.error("Token verification error: %s", e)
        return None

def cognito_authorize_url() -> Optional[str]:
    if not COGNITO_CLIENT_ID or not DOMAIN:
        return None
    params = {
        "client_id": COGNITO_CLIENT_ID,
        "response_type": "code",
        "scope": "openid email profile",
        "redirect_uri": REDIRECT_URI,
    }
    return f"{DOMAIN}/oauth2/authorize?{urllib.parse.urlencode(params)}"

def current_user() -> Optional[Dict[str, Any]]:
    id_token = session.get("id_token")
    if not id_token:
        return None
    return verify_token(id_token)

# ---------- Before request ----------
@app.before_request
def attach_user():
    g.user = current_user()

# ---------- Qdrant helpers ----------
def get_available_collections() -> List[str]:
    return [c.name for c in qdrant.get_collections().collections]

def semantic_search(query_text: str, collections: List[str], limit: int = 5, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
    query_vector = embedding_model.encode(query_text).tolist()
    all_results = []

    for name in collections:
        try:
            hits = qdrant.search(
                collection_name=name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                score_threshold=similarity_threshold,
            )
            for hit in hits:
                result = {
                    "collection": name,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "metadata": {},
                }
                if "metadata" in hit.payload:
                    result["metadata"] = hit.payload["metadata"]
                else:
                    for key in hit.payload:
                        if key != "text":
                            result["metadata"][key] = hit.payload[key]
                all_results.append(result)
        except Exception as e:
            logger.error("Error searching %s: %s", name, e)

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:limit]

# ---------- Gemini generation ----------
def generate_response(query: str, context_chunks: List[Dict], temperature: float = 0.7, chat_history: List[Dict] = None) -> Dict[str, Any]:
    formatted_chunks = []
    for chunk in context_chunks:
        meta = chunk.get("metadata", {})
        parts = [f"Collection: {chunk['collection']}"]
        for field in ("book_title", "publication_year", "author", "doc_type", "source_file"):
            val = meta.get(field)
            if val:
                parts.append(f"{field.replace('_', ' ').title()}: {val}")
        formatted_chunks.append(f"Source [{', '.join(parts)}]: {chunk['text']}")

    context = "\n\n".join(formatted_chunks)

    system_instruction = (
        "You are playing the role of Phyllis Schlafly that answers questions based on the provided context. "
        "If the context doesn't contain relevant information to answer the question, "
        "say that you don't have enough information. "
        "Cite your sources with endnotes at the end of each response, referencing the source metadata provided. "
        "Always maintain Phyllis Schlafly's conservative perspective and voice."
    )

    # Build conversation history
    history_text = ""
    if chat_history:
        for pair in chat_history[-4:]:
            history_text += f"User: {pair.get('query', '')}\nAssistant: {pair.get('response', '')}\n\n"

    prompt = f"{history_text}Context:\n{context}\n\nQuestion: {query}"

    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        text = response.text or "I couldn't generate a response."
        usage = getattr(response, "usage_metadata", None)
        token_info = {
            "input_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_token_count", 0) if usage else 0,
        }
        return {"response": text, "token_info": token_info}
    except Exception as e:
        logger.error("Gemini error: %s", e)
        return {"response": f"Error generating response: {e}", "token_info": {}}

# ---------- Download helpers ----------
def format_conversation_text(messages: List[Dict], include_references: bool = True) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"Conversation Export - {timestamp}\n\n"

    for i, msg in enumerate(messages, 1):
        query = msg.get("query", "")
        response = msg.get("response", "")
        chunks = msg.get("chunks", [])

        text += f"{'=' * 80}\nMessage {i}\n{'=' * 80}\n\n"
        text += f"Question:\n{query}\n\nAnswer:\n{response}\n\n"

        if include_references and chunks:
            text += "Reference Chunks:\n"
            for j, chunk in enumerate(chunks, 1):
                text += f"\nChunk {j}:\n"
                text += f"Collection: {chunk.get('collection', 'N/A')}\n"
                text += f"Text: {chunk.get('text', 'N/A')}\n"
                text += f"Score: {chunk.get('score', 'N/A')}\n"
                metadata = chunk.get("metadata", {})
                if metadata:
                    text += "Metadata:\n"
                    for key, value in metadata.items():
                        text += f"  {key}: {value}\n"
                text += "-" * 80 + "\n"
        text += "\n"

    return text

def create_pdf(messages: List[Dict], include_references: bool = True) -> bytes:
    from xml.sax.saxutils import escape

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("CustomTitle", parent=styles["Heading1"], fontSize=16, spaceAfter=30)
    heading_style = ParagraphStyle("CustomHeading", parent=styles["Heading2"], fontSize=14, spaceAfter=12)
    subheading_style = ParagraphStyle("CustomSubheading", parent=styles["Heading3"], fontSize=12, spaceAfter=8)
    normal_style = styles["Normal"]
    chunk_text_style = ParagraphStyle(
        "ChunkText", parent=normal_style, fontSize=9, leading=11,
        leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=4,
        borderPadding=5, borderWidth=1, borderColor=colors.lightgrey,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements = [Paragraph(f"Conversation Export - {timestamp}", title_style), Spacer(1, 12)]

    for i, msg in enumerate(messages, 1):
        query = msg.get("query", "")
        response = msg.get("response", "")
        chunks = msg.get("chunks", [])

        if i > 1:
            elements.append(Spacer(1, 20))

        elements.append(Paragraph(f"Message {i}", heading_style))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("Question:", subheading_style))
        elements.append(Paragraph(escape(query), normal_style))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Answer:", subheading_style))
        elements.append(Paragraph(escape(response), normal_style))
        elements.append(Spacer(1, 12))

        if include_references and chunks:
            elements.append(Paragraph("Reference Chunks:", subheading_style))
            for j, chunk in enumerate(chunks, 1):
                elements.append(Spacer(1, 8))
                elements.append(Paragraph(f"<b>Chunk {j}</b>", normal_style))
                elements.append(Spacer(1, 4))

                metadata_data = [
                    ["Collection:", chunk.get("collection", "N/A")],
                    ["Score:", str(chunk.get("score", "N/A"))],
                ]
                for key, value in chunk.get("metadata", {}).items():
                    metadata_data.append([f"{key}:", str(value)])

                table = Table(metadata_data, colWidths=[1.5 * inch, 5 * inch])
                table.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 4))
                elements.append(Paragraph(f"<b>Text:</b> {escape(chunk.get('text', 'N/A'))}", chunk_text_style))
                elements.append(Spacer(1, 6))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# ---------- Auth Routes ----------
@app.route("/login")
def login():
    auth_url = cognito_authorize_url()
    if not auth_url:
        return ("Authentication not configured", 500)
    return redirect(auth_url)

@app.route("/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return ("Missing code", 400)
    if not DOMAIN or not COGNITO_CLIENT_ID:
        return ("Authentication not configured", 500)

    token_url = f"{DOMAIN}/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": COGNITO_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "code": code,
    }
    r = http_requests.post(token_url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    if r.status_code != 200:
        return (f"Token exchange failed: {r.text}", 400)

    tokens = r.json()
    session["id_token"] = tokens["id_token"]
    session["access_token"] = tokens.get("access_token")
    return redirect("/")

@app.route("/logout")
def logout():
    session.clear()
    if DOMAIN and COGNITO_CLIENT_ID:
        logout_url = f"{DOMAIN}/logout?{urllib.parse.urlencode({'client_id': COGNITO_CLIENT_ID, 'logout_uri': REDIRECT_URI.rsplit('/callback', 1)[0] + '/'})}"
        return redirect(logout_url)
    return redirect("/")

@app.route("/me")
def me():
    if not g.user:
        return jsonify({"authenticated": False}), 401
    return jsonify({
        "authenticated": True,
        "email": g.user.get("email"),
        "name": g.user.get("name", g.user.get("email", "User")),
        "sub": g.user.get("sub"),
    })

# ---------- App Routes ----------
@app.route("/")
def index():
    collections = get_available_collections()
    return render_template("index.html", collections=collections)

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})

@app.route("/api/query", methods=["POST"])
def query_api():
    if REQUIRE_AUTH and not g.user:
        return jsonify({"error": "Authentication required"}), 401

    data = request.json
    query = data.get("query", "").strip()
    collections = data.get("collections", [])
    chunk_limit = int(data.get("chunk_limit", 5))
    temperature = float(data.get("temperature", 0.7))
    similarity_threshold = float(data.get("similarity_threshold", 0.0))
    chat_history = data.get("chat_history", [])

    if not query:
        return jsonify({"error": "Query is required"}), 400
    if not collections:
        return jsonify({"error": "At least one collection must be selected"}), 400

    chunks = semantic_search(query, collections, limit=chunk_limit, similarity_threshold=similarity_threshold)
    result = generate_response(query, chunks, temperature=temperature, chat_history=chat_history)

    return jsonify({
        "response": result["response"],
        "chunks": chunks,
        "token_info": result.get("token_info", {}),
    })

@app.route("/api/download/<fmt>", methods=["POST"])
def download_conversation(fmt):
    if REQUIRE_AUTH and not g.user:
        return jsonify({"error": "Authentication required"}), 401

    data = request.json
    messages = data.get("messages", [])
    include_references = data.get("include_references", True)

    if fmt not in ("txt", "pdf"):
        return jsonify({"error": "Invalid format. Must be 'txt' or 'pdf'"}), 400
    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "txt":
        content = format_conversation_text(messages, include_references)
        buffer = io.BytesIO(content.encode("utf-8"))
        filename = f"conversation_{timestamp}.txt"
        mimetype = "text/plain"
    else:
        buffer = io.BytesIO(create_pdf(messages, include_references))
        filename = f"conversation_{timestamp}.pdf"
        mimetype = "application/pdf"

    buffer.seek(0)
    return send_file(buffer, mimetype=mimetype, as_attachment=True, download_name=filename)

# ---------- Main ----------
if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not set")
    try:
        colls = get_available_collections()
        logger.info("Collections: %s", colls)
    except Exception as e:
        logger.error("Qdrant connection error: %s", e)
    app.run(debug=True, host="0.0.0.0", port=5000)
