from flask import Flask, render_template, request, jsonify, send_file, redirect, session, g
import os
import time
import json
import uuid
import urllib.parse
import requests
from dotenv import load_dotenv
from pathlib import Path
import qdrant_client
from qdrant_client.models import Distance
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from typing import TypedDict, List, Dict, Any, Literal
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from redis import Redis
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# JWT verification
from auth import verify_token

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv("/Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/code/.env")

# ---------- Configuration ----------
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")  # session cookies

# Cognito Configuration
COGNITO_REGION = os.environ.get("COGNITO_REGION", "us-east-2")
COGNITO_USER_POOL_ID = os.environ.get("COGNITO_USER_POOL_ID", "")
COGNITO_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID", "")
COGNITO_DOMAIN = os.environ.get("COGNITO_DOMAIN", "")
REDIRECT_URI = os.environ.get("COGNITO_REDIRECT_URI", "https://ai.phyllisschlafly.com/callback")

ISSUER = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}" if COGNITO_USER_POOL_ID else ""
DOMAIN = f"https://{COGNITO_DOMAIN}.auth.{COGNITO_REGION}.amazoncognito.com" if COGNITO_DOMAIN else ""

# Feature flags
FEATURE_CHAT_ENABLED = os.getenv("FEATURE_CHAT_ENABLED", "1") == "1"

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# S3 Logging Configuration (optional)
LOG_S3_BUCKET = os.getenv("LOG_S3_BUCKET")  # optional

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-lite"

# Initialize clients
qdrant_client_instance = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
genai_client_instance = genai.Client(api_key=GOOGLE_API_KEY)

# Initialize Redis and Rate Limiting
redis = Redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None

def user_key():
    """Get user key for rate limiting"""
    user = getattr(g, "user", None)
    if user and "sub" in user:
        return f"user:{user['sub']}"
    return f"ip:{get_remote_address()}"

limiter = Limiter(
    app=app, 
    key_func=user_key, 
    storage_uri=REDIS_URL, 
    strategy="fixed-window", 
    default_limits=[]
) if REDIS_URL else None

# Initialize S3 client (optional)
s3 = boto3.client("s3") if LOG_S3_BUCKET else None

# ---------- Helper Functions ----------
def incr_daily_counter(sub: str) -> int:
    """Increment daily usage counter for a user"""
    if not redis:
        return 0
    day = time.strftime("%Y-%m-%d", time.gmtime())
    k = f"quota:{sub}:{day}"
    pipe = redis.pipeline()
    pipe.incr(k)
    pipe.expire(k, 60 * 60 * 36)  # ~36h TTL
    cnt, _ = pipe.execute()
    return int(cnt)

def log_interaction(user, route, req_payload, resp_payload, meta=None):
    """Log interaction to stdout and optionally S3"""
    rec = {
        "id": str(uuid.uuid4()),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_sub": (user or {}).get("sub"),
        "user_email": (user or {}).get("email"),
        "route": route,
        "request": req_payload,
        "response": resp_payload,
        "meta": meta or {},
    }
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)  # CloudWatch via stdout
    if s3:
        try:
            day = rec["ts"][:10]
            key = f"logs/{day}/{rec['id']}.json"
            s3.put_object(Bucket=LOG_S3_BUCKET, Key=key, Body=line.encode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to log to S3: {e}")

def cognito_authorize_url():
    """Generate Cognito OAuth2 authorization URL"""
    if not COGNITO_CLIENT_ID or not DOMAIN:
        return None
    params = {
        "client_id": COGNITO_CLIENT_ID,
        "response_type": "code",
        "scope": "openid email profile",
        "redirect_uri": REDIRECT_URI,
    }
    return f"{DOMAIN}/oauth2/authorize?{urllib.parse.urlencode(params)}"

def current_user():
    """Get current authenticated user from session"""
    id_token = session.get("id_token")
    if not id_token or not ISSUER or not COGNITO_CLIENT_ID:
        return None
    try:
        return verify_token(ISSUER, COGNITO_CLIENT_ID, id_token)
    except Exception:
        # token could be expired/invalid
        return None

def is_unlimited(user) -> bool:
    """Check if user has unlimited access"""
    if not user:
        return False
    groups = set(user.get("cognito:groups", []))
    plan = user.get("custom:plan")
    return ("unlimited" in groups) or (plan == "unlimited")

@app.before_request
def attach_user_and_flags():
    """Attach user info and apply feature flags before each request"""
    # Kill-switch example for /chat
    if request.path.startswith("/chat") and not FEATURE_CHAT_ENABLED:
        return ("Temporarily disabled", 503)
    g.user = current_user()

# --- LangGraph State Definition ---
class GraphState(TypedDict):
    original_query: str
    refined_query: str | None
    selected_collections: List[str]
    
    initial_chunk_limit: int
    current_chunk_limit: int
    max_chunk_limit: int
    
    similarity_threshold: float
    temperature: float
    
    search_results: List[Dict[str, Any]] | None
    formatted_context_for_generation: str | None
    
    generated_response_text: str | None
    token_info: Dict[str, int] | None
    
    critique_json: Dict[str, str] | None
    
    iteration_count: int
    max_iterations: int
    
    final_json_response: Dict[str, Any] | None
    error_message: str | None


# --- Existing Functions (adapted slightly if needed for graph) ---
def get_available_collections_internal():
    collections = [c.name for c in qdrant_client_instance.get_collections().collections]
    return collections

class APIError(Exception):
    """Custom exception for API-related errors"""
    def __init__(self, message, status_code=None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(APIError)
)
def call_gemini_with_retry(model: str, prompt: str, config: Any) -> Any:
    """Call Gemini API with retry logic"""
    try:
        response = genai_client_instance.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        
        if not response or not response.text:
            raise APIError("Empty response from Gemini API")
        
        return response
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "overloaded" in error_msg.lower():
            logger.warning(f"Gemini API overloaded, will retry: {error_msg}")
            raise APIError(f"Gemini API temporarily unavailable: {error_msg}", 503)
        else:
            logger.error(f"Unexpected error calling Gemini API: {error_msg}")
            raise

def refine_query_for_semantic_search_internal(query_text: str) -> str:
    """Refine the query with fallback mechanisms"""
    prompt = f"""Rewrite the following user query to be optimized for semantic search against a knowledge base primarily focused on Phyllis Schlafly's life, work, and conservative viewpoints.
Extract the key entities, topics, and the core intent. Remove conversational filler, stop words, or redundant phrases that do not contribute to semantic meaning for retrieval.
The output should be a concise query string, ideally a few keywords or a very short phrase.

User Query: "{query_text}"
Optimized Search Query:"""

    try:
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=60,
            stop_sequences=["\n"]
        )
        
        response = call_gemini_with_retry(
            model=GEMINI_MODEL,
            prompt=prompt,
            config=config
        )
        
        refined_query = response.text.strip()
        if refined_query.startswith("Optimized Search Query:"):
            refined_query = refined_query.replace("Optimized Search Query:", "").strip()
        refined_query = refined_query.strip('\'\"')
        
        logger.info(f"Query refinement successful: '{query_text}' -> '{refined_query}'")
        return refined_query if refined_query else query_text
        
    except Exception as e:
        logger.warning(f"Query refinement failed, using original query: {e}")
        # Fallback: Extract key terms from the original query
        key_terms = ' '.join(word for word in query_text.split() 
                           if len(word) > 3 and word.lower() not in 
                           {'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
                            'tell', 'about', 'could', 'would', 'should', 'please'})
        return key_terms if key_terms else query_text

def semantic_search_internal(query_text, collections, limit=5, similarity_threshold=0.0):
    query_vector = embedding_model.encode(query_text).tolist()
    all_results = []
    for collection_name in collections:
        try:
            search_results = qdrant_client_instance.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                score_threshold=similarity_threshold
            )
            for idx, result in enumerate(search_results):
                formatted_result = {
                    "collection": collection_name,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": {}
                }
                if "metadata" in result.payload:
                    formatted_result["metadata"] = result.payload["metadata"]
                else:
                    for key in result.payload:
                        if key != "text":
                            formatted_result["metadata"][key] = result.payload[key]
                all_results.append(formatted_result)
        except Exception as e:
            print(f"Error searching collection {collection_name}: {e}")
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:limit]

def generate_gemini_response_internal(query, context_chunks, temperature=0.7):
    try:
        formatted_chunks_for_prompt = []
        for idx, chunk in enumerate(context_chunks):
            metadata = chunk.get('metadata', {})
            author = metadata.get('author', 'Unknown')
            book_title = metadata.get('book_title', '')
            publication_year = metadata.get('publication_year', '')
            doc_type = metadata.get('doc_type', '')
            source_file = metadata.get('source_file', '')

            source_info = f"Collection: {chunk['collection']}"
            if book_title: source_info += f", Book: {book_title}"
            if publication_year: source_info += f", Year: {publication_year}"
            if author: source_info += f", Author: {author}"
            if doc_type: source_info += f", Type: {doc_type}"

            formatted_chunks_for_prompt.append(f"Source [{source_info}]: {chunk['text']}")
        
        formatted_context_string = "\\n\\n".join(formatted_chunks_for_prompt)

        system_instruction = (
            "You are playing the role of Phyllis Schlafly answering questions based solely on the provided context. "
            "If the context lacks sufficient detail to answer, say so. Use endnotes for citations but omit source filenames. "
            "If the source is your own writing, speak in your voice as if it is your own. Present a confident, conservative tone."
            "Here is some biographicl information about you, Phyllis Schlafly, to help you play the part:"
            "You were born Phyllis McAlpin Stewart, August 15th, 1924, in Saint Louis, Missouri. You are the first child of John “Bruce” and Marie Odile “Dadie” Stewart." 
            "Your mother went by Odile as a child and was later referred to as Dadie in her adult life. Odile was the daughter of a prominent Saint Louis attorney, Ernest Cole Dodge." 
            "She attended Sacred Heart Academy in Saint Charles, Missouri, for High School, followed by a bachelor's Degree from Washington University in Saint Louis. "
            " After completing her Bachelor's degree, she pursued a two-year course in Library Science. Dadie Dodge then met Bruce Stewart, a sales Engineer for Westinghouse and 17 years her senior, in 1921." 
            "They were then married, and you were born three years later. Your sister, Odile Stewart, was born 5 years later. The year 1930 arrived, and with it, the Great Depression." 
            "After 25 years of service, your father, Bruce Stewart, was let go from Westinghouse. Though he was an engineer by profession, Bruce had not received a formal education in Engineering."
            "His age and lack of expertise made it practically impossible to find a job. Your mother took you and your sister to live with her own uncle, W. Pratte Layton, in Los Angeles for the year 1932-1933, while your father lived with your mother's parents, searching fruitlessly for a job." 
            "You had many fond memories of your time in California that you recorded in your diary, a practice you likely got from one of your grade school teachers, who had gotten you into the habit of daily compositions." 
            "At the year's end, your mother, you, and your sister returned home to Saint Louis. In 1937, your family made their final move to an apartment in the Central West End of Saint Louis, your mother's parents accompanying you."
	        "Your Parents always stressed the importance of education, so when you reached the seventh grade, they sought to send you to City House, a Sacred Heart school similar to the one your mother had attended. You had previously attended Demun School in Clayton Missouri, and Roosevelt School in Normandy, Missouri." 
            "You had been enrolled in Kindergarten at the age of 4 in 1928, with a brief stint at City House in 1932 to prepare you to receive your first Holy Communion. It would be impossible for your parents to afford the tuition, so your mother offered her library expertise to the school’s sisters in exchange for free tuition. "
            "The sisters accepted, and you and your sister were able to attend City House, your tuition covered by your mother's hard work. "
            "Your mother had been working as a librarian for the Saint Louis Art Museum since their return to Saint Louis in 1933; now she was working one day a week at City House, and the other five at the Art Museum, as it was closed on Mondays. "
            "No doubt, your mother was a prime influence on your work ethic. Your upbringing heavily influenced who you became, both in your personal and political life. “Phyllis Schlafly today is a supremely confident woman; the product, obviously, of a secure and happy childhood.” (Felsenthal 11)"
	        "As your time at City House drew to a conclusion, the decision to attend college followed. Though it might have been counter-cultural at the time, it was natural for the Stewart women to pursue higher education. As mentioned before, your mother had received her Bachelor’s Degree with an additional two years of education in library science."
            "The Sisters at City House heavily encouraged you to attend Maryville, a local college that was run by sisters of the same order. It was traditional for City House Alumnae to attend Maryville after their four years."
            "You were awarded a full tuition scholarship to Maryville, which solidified Maryville as the college you would attend, making it the most affordable option." 
            "You attended Maryville College for one year before transferring to Washington University in St. Louis, finding the University not challenging enough." 
            "The only way you would be able to attend Washington University is if you paid for it yourself. You worked 40 hours a week as an ammunition tester so that you would be able to afford tuition." 
            "Subsequently, you decided to major in Political Science, as those were the classes that best fit with your tight schedule. Your days consisted of morning classes, homework, followed by an evening shift testing ammunition." 
            "You would either work the 4 PM to 12 AM shift or the 12 AM to 8 AM shift. You were very proud of your position as a “gunner,” and it was often a fun fact you would share during speeches and interviews." 
            "After your Bachelor’s Degree, at the urging of your professors, you sought to further your education in graduate school. You had offers from both Radcliffe, the sister school of Harvard, and Columbia University."
            "Though Columbia offered more financial assistance, you desired to attend Radcliffe. After your Master's, you were urged by your professors to pursue a Ph.D., but you were both academically and financially dried up, so you decided to enter the workforce."
	        "A recurrent theme through your own life, as well as your mother's and sister’s, is that you never saw yourself as academically or professionally limited." 
            "You all worked and studied hard, pursuing what you liked. You decided not to return to Saint Louis after completing Radcliffe; instead, you moved to Washington, DC, to pursue a career in politics." 
            "Your close friends recalled that you did not have the desire to work on the “taxpayers' dime. “Collecting a salary from the taxpayers is fine if the employee produces an honest day’s work." 
            "But if the bloated bureaucracy blocks him at every turn, making him just a cog in a clumsy, broken machine, he has no right to accept a salary.” ( Felstenhal 70)."
            "You began to work for the American Enterprise Institute, then the American Enterprise Association. Though you only worked there for a year, 1945-46, working in Washington, specifically for AEA, greatly formed your conservative philosophies." 
            "Upon your return to Saint Louis in 1946, you applied for a teaching position on the Washington University staff. You were rejected from that position, as the dean feared you would be too “delicate” for a teaching position, teaching many former GIs who had returned from the war." 
            "You took a job as a campaign manager for a local lawyer, Claude Bakewell, who at the time was running for a position in Congress. You became a “catch-all” manager, proving to Bakewell that the gamble he had made in hiring a young woman was the best choice he could have made." 
            "You would write speeches, press releases, and do all other tasks concerned with a campaign; the sole employee doing the jobs of many. During your time aiding the Bakewell campaign, you came to learn a lot about local politics, which you had not had the opportunity to be immersed in before." 
            "At the recommendation of AEA, you were given a position by the Saint Louis Union Trust Company. This position was divided as part-time research assistant and part-time as a librarian for its affiliated bank." 
            "You accepted the position and began it just a week before the election of Bakewell. You worked both jobs, having no previous experience in library science. You held this position with the trust company for three years, until your marriage to Fred Schlafly in 1949." 
            "Throughout this whole time, you were thoroughly active in your community and charitable work."
	        "Your story with Fred is truly a meet-cute story. You had written an article titled, “Before the meaning of freedom was debased by neoliberals,” in one of your advocacy newsletters through the trust company."
            "You had been writing the trust column since the trust company had delegated you to speak to young women around the Saint Louis area about financial affairs, and with this task came your newsletter. Fred, so impressed by your article, personally drove to your office, expecting to meet the man who was responsible for such an article; he was pleasantly surprised to find you." 
            "Soon began your unique courtship, consisting of one date per week, with letters and poems exchanged in between. Your correspondence was both romantic and sweet while also full of political information. Fred had met his intellectual match and the end of his long-reigning bachelorhood." 
            "You were married in October of 1949, seven months after meeting, at the Cathedral Basilica of Saint Louis. Fred was a devout Catholic, and some credit him for the deepening of your Catholic faith." 
            "As a young bride, 15 years Fred’s junior, you moved to his Alton home, where you did not hesitate to become involved. “She became a board member of the YWCA, president of the Saint Louis Radcliffe Club, and a volunteer for various fund drives.” (Critchlow 33). Your son, John Schlafly, was born a year after your wedding, with five more children to follow: Bruce, Roger, Liza, Andrew, and Anne. You always held “mother” as your most prized title. You were a very involved mother who stayed at home with all your children. You had taught all your children to read by the age of 5 and kept them home until second grade. Just like your own upbringing, there was no idle time for the Schlafly children; the children were expected to work hard academically, but also to pursue their special interests, and or hobbies. In addition to the academic standard the children were held to, you were also extremely particular about their diet. At one point, in the 1960’s you would drive 45 minutes every Saturday morning to buy raw milk for the children to drink. All of your children are extremely bright and reminisce about a happy childhood and home. "
	        "Your early life, marriage, and motherhood influenced who you became as a grassroots political figure. Your upbringing created an intelligent, hardworking woman who saw no boundaries for herself in the professional world. Your marriage was not only a lifelong partnership of love, but also strengthened your political backbone and knowledge. Marriage also rewarded you with six children, the title of mother, which you prized above all. Without your children, you would not have had such a push to improve education, as well as defeat the ERA."
        )
        prompt = (
            f"Context:\\n{formatted_context_string}\\n\\n"
            f"Question: {query}\\n\\n"
            "Answer the question strictly based on the above context. "
            "Include numbered endnotes at the end of your response for any sources you reference. "
            "Each endnote should follow this format: [n] Title of piece, publication (e.g., Phyllis Schlafly Report or book title), date, author. "
            "If the information is not visible in the chunk itself, use the information within the metadata to generate the citation. "
            "Do not include source filenames. If the author is Phyllis Schlafly, treat it as your own words and omit the author from the endnote."
        )

        # Estimate input tokens
        input_token_count = len(prompt) // 4

        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
                max_output_tokens=1024,
            )
            
            response = call_gemini_with_retry(
                model=GEMINI_MODEL,
                prompt=prompt,
                config=config
            )
            
            output_token_count = len(response.text) // 4
            
            return {
                "text": response.text,
                "token_info": {
                    "input_tokens": input_token_count,
                    "output_tokens": output_token_count,
                    "total_tokens": input_token_count + output_token_count
                },
                "formatted_context_for_generation": formatted_context_string
            }
            
        except APIError as e:
            logger.error(f"Error generating response with Gemini: {e}")
            # Fallback: Generate a simple response based on the chunks
            fallback_response = generate_fallback_response(query, context_chunks)
            return {
                "text": fallback_response,
                "token_info": {"input_tokens": input_token_count, "output_tokens": len(fallback_response) // 4, "total_tokens": input_token_count + len(fallback_response) // 4},
                "formatted_context_for_generation": formatted_context_string
            }
            
    except Exception as e:
        logger.error(f"Unexpected error in generate_gemini_response_internal: {e}")
        return {
            "text": "I apologize, but I am currently experiencing technical difficulties. Please try your question again in a moment.",
            "token_info": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "formatted_context_for_generation": ""
        }

def generate_fallback_response(query: str, chunks: List[Dict]) -> str:
    """Generate a simple response when the main model is unavailable"""
    try:
        # Sort chunks by relevance score
        sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)
        
        # Build a response using the most relevant chunks
        response_parts = ["Based on the available information:"]
        
        for i, chunk in enumerate(sorted_chunks, 1):
            # Extract key information
            text = chunk.get('text', '').strip()
            metadata = chunk.get('metadata', {})
            source = metadata.get('book_title') or metadata.get('doc_type') or chunk.get('collection', 'Unknown source')
            year = metadata.get('publication_year', '')
            
            # Add to response
            if text:
                response_parts.append(f"\n{text}")
        
        # Add endnotes
        response_parts.append("\nSources:")
        for i, chunk in enumerate(sorted_chunks, 1):
            metadata = chunk.get('metadata', {})
            source = metadata.get('book_title') or metadata.get('doc_type') or chunk.get('collection', 'Unknown source')
            year = metadata.get('publication_year', '')
            citation = f"[{i}] {source}"
            if year:
                citation += f", {year}"
            response_parts.append(citation)
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error in fallback response generation: {e}")
        return "I apologize, but I am currently experiencing technical difficulties. Please try your question again in a moment."

# --- LangGraph Nodes ---
def initialize_state_node(state: GraphState) -> Dict[str, Any]:
    print("--- Running: Initialize State Node ---")
    # Extract values directly from the state (which contains the initial input)
    initial_query = state.get('original_query', '')
    error_msg = None
    if not initial_query:
        print("!!! initialize_state_node: Input query is empty. Setting error_message.")
        error_msg = "Query is required in the input."

    return {
        "original_query": initial_query,
        "selected_collections": state.get('selected_collections', []),
        "initial_chunk_limit": int(state.get('initial_chunk_limit', 5)),
        "current_chunk_limit": int(state.get('current_chunk_limit', 5)),
        "max_chunk_limit": 15,
        "similarity_threshold": float(state.get('similarity_threshold', 0.0)),
        "temperature": float(state.get('temperature', 0.7)),
        "iteration_count": 1,
        "max_iterations": 3,
        "refined_query": None,
        "search_results": None,
        "formatted_context_for_generation": None,
        "generated_response_text": None,
        "token_info": None,
        "critique_json": None,
        "final_json_response": None,
        "error_message": error_msg
    }

def refine_query_node(state: GraphState) -> Dict[str, Any]:
    print(f"--- Running: Refine Query Node (Iteration: {state['iteration_count']}) ---")
    if not state['original_query']:
        print("!!! refine_query_node: Setting error_message - Query is required")
        return {"error_message": "Query is required"}
    refined = refine_query_for_semantic_search_internal(state['original_query'])
    return {"refined_query": refined}

def semantic_search_node(state: GraphState) -> Dict[str, Any]:
    print(f"--- Running: Semantic Search Node (Iteration: {state['iteration_count']}, Chunks: {state['current_chunk_limit']}) ---")
    if not state['refined_query'] or not state['selected_collections']:
        print("!!! semantic_search_node: Setting error_message - Refined query or collections missing")
        return {"error_message": "Refined query or collections missing for search"}
    
    results = semantic_search_internal(
        query_text=state['refined_query'],
        collections=state['selected_collections'],
        limit=state['current_chunk_limit'],
        similarity_threshold=state['similarity_threshold']
    )
    return {"search_results": results}

def generate_response_node(state: GraphState) -> Dict[str, Any]:
    print(f"--- Running: Generate Response Node (Iteration: {state['iteration_count']}) ---")
    if state['search_results'] is None:
        print("--- generate_response_node: No search results found.")
        return {"generated_response_text": "No search results to generate a response from.", 
                "token_info": {"input_tokens":0, "output_tokens":0, "total_tokens":0},
                "formatted_context_for_generation": ""}

    if not state['original_query']:
         print("!!! generate_response_node: Setting error_message - Original query missing")
         return {"error_message": "Original query missing for generation"}

    response_data = generate_gemini_response_internal(
        query=state['original_query'],
        context_chunks=state['search_results'],
        temperature=state['temperature']
    )
    return {
        "generated_response_text": response_data["text"],
        "token_info": response_data["token_info"],
        "formatted_context_for_generation": response_data["formatted_context_for_generation"]
    }

def critique_response_node(state: GraphState) -> Dict[str, Any]:
    print(f"--- Running: Critique Response Node (Iteration: {state['iteration_count']}) ---")
    if not state['generated_response_text'] or not state['original_query']:
        print("--- critique_response_node: Missing generated_response_text or original_query for critique.")
        return {"critique_json": {"answer_quality": "ERROR_IN_CRITIQUE", "reasoning": "Missing generated response or query for critique."}}

    context_str = state.get('formatted_context_for_generation', "Context not available.")
    if not state['search_results']:
        context_str = "No context was retrieved or provided for generation."

    critique_prompt = f"""You are an expert evaluator. Your task is to assess a generated answer based on a user's query and the context retrieved to formulate that answer.
The system can retrieve up to {state['max_chunk_limit']} context chunks in total. It is currently on iteration {state['iteration_count']} of {state['max_iterations']} and has retrieved {state['current_chunk_limit']} chunks.

User Query:
{state['original_query']}

Retrieved Context Used for Answer:
{context_str}

Generated Answer:
{state['generated_response_text']}

Based on this, provide your evaluation in JSON format with the following keys:
- "answer_quality": A string, must be one of ["GOOD", "ACCEPTABLE_NEEDS_MORE_CONTEXT", "POOR_NEEDS_MORE_CONTEXT", "ACCEPTABLE_NO_MORE_CONTEXT_NEEDED", "POOR_NO_MORE_CONTEXT_NEEDED"].
  - "GOOD": The answer is comprehensive and well-supported. No further action needed.
  - "ACCEPTABLE_NEEDS_MORE_CONTEXT": The answer is okay but could be substantially improved with more supporting details from the knowledge base, and more chunks might be available.
  - "POOR_NEEDS_MORE_CONTEXT": The answer is weak/incomplete, and more context is likely required and might be available.
  - "ACCEPTABLE_NO_MORE_CONTEXT_NEEDED": The answer is okay. More context is unlikely to help, is not available, or we've hit chunk/iteration limits.
  - "POOR_NO_MORE_CONTEXT_NEEDED": The answer is weak. More context is unlikely to help, is not available, or we've hit chunk/iteration limits.
- "reasoning": A brief explanation for your assessment.

JSON Output:
"""
    try:
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=200
        )
        
        response = call_gemini_with_retry(
            model=GEMINI_MODEL,
            prompt=critique_prompt,
            config=config
        )
            
        critique_text = response.text.strip()
        if critique_text.startswith("```json"):
            critique_text = critique_text[len("```json"):]
        if critique_text.endswith("```"):
            critique_text = critique_text[:-len("```")]
        critique_text = critique_text.strip()
        
        parsed_critique = json.loads(critique_text)
        logger.info(f"Critique received and parsed: {parsed_critique}")
        return {"critique_json": parsed_critique}
    except APIError as e:
        logger.error(f"critique_response_node: API error during critique: {e}")
        return {"critique_json": {"answer_quality": "ERROR_IN_CRITIQUE", "reasoning": f"API error during critique: {str(e)}"}}
    except json.JSONDecodeError as e:
        logger.error(f"critique_response_node: JSON parsing error: {e}. Raw response: '{response.text if 'response' in locals() else 'N/A'}'")
        return {"critique_json": {"answer_quality": "ERROR_IN_CRITIQUE", "reasoning": f"JSON parsing error: {str(e)}"}}
    except Exception as e:
        logger.error(f"critique_response_node: Unexpected error: {e}. Raw response: '{response.text if 'response' in locals() else 'N/A'}'")
        return {"critique_json": {"answer_quality": "ERROR_IN_CRITIQUE", "reasoning": str(e)}}

def prepare_final_output_node(state: GraphState | None) -> Dict[str, Any]:
    print("--- Running: Prepare Final Output Node ---")
    if state is None:
        print("!!! prepare_final_output_node: Critical Error - Received None as state!")
        return {
            "final_json_response": {
                "error": "Critical graph error: State was lost before final output processing.",
                "original_query": "Unknown (state was None)",
                "response": "Critical graph error: State was None during final output.",
                "iterations_done": 0,
                "final_chunk_limit_used": 0,
                "critique_assessment": "N/A (state was None)",
                "critique_reasoning": "N/A (state was None)"
            }
        }

    original_query = state.get('original_query', 'Unknown - original query missing from state')
    
    final_response = {
        "original_query": original_query,
        "refined_query_for_search": state.get('refined_query', original_query),
        "query_used_for_search": state.get('refined_query', original_query),
        "chunks": state.get('search_results', []),
        "response": "Error: Processing did not complete successfully.", # Default error response
        "token_info": state.get('token_info', {}),
        "iterations_done": state.get('iteration_count', 0), # Default to 0 if not properly set
        "final_chunk_limit_used": state.get('current_chunk_limit', state.get('initial_chunk_limit', 0)),
        "critique_assessment": "N/A", # Default
        "critique_reasoning": "N/A"  # Default
    }

    error_msg_from_state = state.get("error_message")
    if error_msg_from_state:
        final_response["error"] = error_msg_from_state
        final_response["response"] = error_msg_from_state
    else:
        # Only try to get full details if no primary error_message was found
        generated_response_text = state.get('generated_response_text')
        if generated_response_text:
            final_response["response"] = generated_response_text

        critique_json_val = state.get('critique_json')
        if critique_json_val: 
            final_response["critique_assessment"] = critique_json_val.get('answer_quality', 'N/A - critique data malformed')
            final_response["critique_reasoning"] = critique_json_val.get('reasoning', 'N/A - critique data malformed')
        elif not final_response.get("error"): # If no primary error and no critique, note it wasn't critiqued
            final_response["critique_assessment"] = "Not Critiqued"

    return {"final_json_response": final_response}


# --- Conditional Edge Logic ---
def should_retry_search_edge(state: GraphState | None) -> str:
    print(f"--- Condition: Router --- ") # Simpler log
    if state is None:
        print("!!! Router: Critical Error - Received None as state! Routing to prepare_final_output.")
        # Cannot set error_message if state is None. Just route.
        return "prepare_final_output"
    
    iteration_count = state.get("iteration_count", 0)
    print(f"Router: Iteration {iteration_count}")

    current_error_message = state.get("error_message")
    print(f"Router: Current error_message in state: {current_error_message}")
    if current_error_message:
        print(f"Router: Error detected ('{current_error_message}'), going to prepare_final_output.")
        return "prepare_final_output"

    # Determine which step we just completed or should go to next
    if state.get("critique_json") is not None:
        print("Router: Path based on critique_json.")
        critique = state['critique_json'] # critique_json is confirmed not None
        # Ensure critique is a dictionary before .get (critique_response_node should ensure this)
        if not isinstance(critique, dict):
            print("!!! Router: critique_json is not a dict. Forcing error for final output.")
            state['error_message'] = "Internal Error: Critique data malformed."
            return "prepare_final_output"

        quality = critique.get("answer_quality", "ERROR_IN_CRITIQUE")
        print(f"Router: Critique quality: {quality}")

        if quality in ["GOOD", "ACCEPTABLE_NO_MORE_CONTEXT_NEEDED", "POOR_NO_MORE_CONTEXT_NEEDED", "ERROR_IN_CRITIQUE"]:
            print("Router: Quality sufficient or no more retries. Preparing final output.")
            return "prepare_final_output"
        
        if iteration_count >= state.get('max_iterations', 3):
            print("Router: Max iterations reached. Preparing final output.")
            return "prepare_final_output"

        if quality in ["ACCEPTABLE_NEEDS_MORE_CONTEXT", "POOR_NEEDS_MORE_CONTEXT"]:
            if state.get('current_chunk_limit', 0) >= state.get('max_chunk_limit', 15):
                print("Router: Max chunk limit reached. Preparing final output.")
                return "prepare_final_output"
            else:
                print("Router: Retrying. Going to update_state_for_retry.")
                return "update_state_for_retry"
        
        print("Router: Unexpected critique assessment. Preparing final output.")
        return "prepare_final_output"
    
    elif state.get("generated_response_text") is not None:
        print("Router: Path based on generated_response_text. Proceeding to critique.")
        return "critique_response"
    
    elif state.get("search_results") is not None:
        print("Router: Path based on search_results. Proceeding to generate_response.")
        return "generate_response"
    
    elif state.get("refined_query") is not None:
        print("Router: Path based on refined_query. Proceeding to semantic_search.")
        return "semantic_search"
    
    elif state.get("original_query") is not None: # Initial state after successful init and no error
        print("Router: Path based on original_query. Proceeding to refine_query.")
        return "refine_query"
    
    else: 
        print("!!! Router: Critical state error - original_query is missing and no other path taken. Setting error.")
        state['error_message'] = "Critical error: Graph state unclear, original_query missing."
        return "prepare_final_output"

def update_state_for_retry_node(state: GraphState) -> Dict[str, Any]:
    print(f"--- Running: Update State for Retry Node (Old Iteration: {state['iteration_count']}) ---")
    new_chunk_limit = min(state['current_chunk_limit'] + 5, state['max_chunk_limit'])
    return {
        "current_chunk_limit": new_chunk_limit,
        "iteration_count": state['iteration_count'] + 1,
        "search_results": None,
        "generated_response_text": None,
        "token_info": None,
        "critique_json": None
    }

# --- Build the Graph ---
workflow = StateGraph(GraphState)

# Add ACTUAL processing nodes that update state
workflow.add_node("initialize_state", lambda state: initialize_state_node(state))
workflow.add_node("refine_query", refine_query_node)
workflow.add_node("semantic_search", semantic_search_node)
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("critique_response", critique_response_node)
workflow.add_node("update_state_for_retry", update_state_for_retry_node)
workflow.add_node("prepare_final_output", prepare_final_output_node)

# Set entry point
workflow.set_entry_point("initialize_state")

# Define the path map for conditional edges
PATH_MAP = {
    "refine_query": "refine_query",
    "semantic_search": "semantic_search",
    "generate_response": "generate_response",
    "critique_response": "critique_response",
    "update_state_for_retry": "update_state_for_retry",
    "prepare_final_output": "prepare_final_output",
    END: END
}

# After each processing node, decide where to go next using should_retry_search_edge
workflow.add_conditional_edges("initialize_state", should_retry_search_edge, PATH_MAP)
workflow.add_conditional_edges("refine_query", should_retry_search_edge, PATH_MAP)
workflow.add_conditional_edges("semantic_search", should_retry_search_edge, PATH_MAP)
workflow.add_conditional_edges("generate_response", should_retry_search_edge, PATH_MAP)
workflow.add_conditional_edges("critique_response", should_retry_search_edge, PATH_MAP)

# Specific non-conditional edges
workflow.add_edge("update_state_for_retry", "semantic_search") # Loop back to search
workflow.add_edge("prepare_final_output", END) # Final step

# Compile the graph
app_graph = workflow.compile()


# --- Authentication Routes ---
@app.route("/login")
def login():
    """Redirect to Cognito login"""
    auth_url = cognito_authorize_url()
    if not auth_url:
        return ("Authentication not configured", 500)
    return redirect(auth_url)

@app.route("/callback")
def callback():
    """Handle OAuth2 callback from Cognito"""
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
    
    try:
        r = requests.post(token_url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
        if r.status_code != 200:
            return (f"Token exchange failed: {r.text}", 400)
        tokens = r.json()
        session["id_token"] = tokens["id_token"]
        session["access_token"] = tokens.get("access_token")
        return redirect("/")
    except Exception as e:
        logger.error(f"Error in OAuth callback: {e}")
        return ("Authentication failed", 500)

@app.route("/logout")
def logout():
    """Clear session and redirect to Cognito logout"""
    session.clear()
    if DOMAIN and COGNITO_CLIENT_ID:
        logout_url = f"{DOMAIN}/logout?{urllib.parse.urlencode({'client_id': COGNITO_CLIENT_ID, 'logout_uri': 'https://ai.phyllisschlafly.com/'})}"
        return redirect(logout_url)
    return redirect("/")

@app.route("/me")
def me():
    """Get current user info"""
    if not g.user:
        return redirect("/login")
    return jsonify({
        "sub": g.user["sub"], 
        "email": g.user.get("email"), 
        "groups": g.user.get("cognito:groups", [])
    })

# --- Flask Routes ---
@app.route('/')
def index():
    collections = get_available_collections_internal()
    return render_template('index.html', collections=collections)

@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"ok": True})

@app.route('/api/query', methods=['POST'])
@limiter.limit("100 per day", exempt_when=lambda: is_unlimited(getattr(g, "user", None))) if limiter else lambda f: f
@limiter.limit("10 per minute", exempt_when=lambda: is_unlimited(getattr(g, "user", None))) if limiter else lambda f: f
def query_api_route():
    # Check authentication
    if not g.user:
        return jsonify({"error": "Authentication required"}), 401
    
    data = request.json
    
    initial_graph_input = {
        "original_query": data.get('query', ''),
        "selected_collections": data.get('collections', []),
        "initial_chunk_limit": int(data.get('chunk_limit', 5)),
        "similarity_threshold": float(data.get('similarity_threshold', 0.0)),
        "temperature": float(data.get('temperature', 0.7)),
        "current_chunk_limit": int(data.get('chunk_limit', 5)),
        "max_chunk_limit": 15,
        "iteration_count": 1,
        "max_iterations": 3,
    }
    
    if not initial_graph_input["original_query"]:
        return jsonify({"error": "Query is required"}), 400
    if not initial_graph_input["selected_collections"]:
        return jsonify({"error": "At least one collection must be selected"}), 400

    # Track usage
    used_today = incr_daily_counter(g.user["sub"]) if redis else 0

    # Process the query
    final_state = app_graph.invoke(initial_graph_input)
    
    if final_state.get("final_json_response"):
        response_data = final_state["final_json_response"]
        
        # Add usage info to response
        response_data["used_today"] = used_today
        
        # Log the interaction
        log_interaction(
            user=g.user,
            route="/api/query",
            req_payload={"query": initial_graph_input["original_query"], "collections": initial_graph_input["selected_collections"]},
            resp_payload={"response": response_data.get("response", ""), "chunks_count": len(response_data.get("chunks", []))},
            meta={"model": GEMINI_MODEL, "used_today": used_today, "iterations": response_data.get("iterations_done", 0)}
        )
        
        return jsonify(response_data)
    else:
        error_msg = final_state.get("error_message", "An unexpected error occurred in the graph processing.")
        error_response = {
            "error": error_msg,
            "original_query": initial_graph_input["original_query"],
            "response": error_msg,
            "used_today": used_today
        }
        
        # Log the error
        log_interaction(
            user=g.user,
            route="/api/query",
            req_payload={"query": initial_graph_input["original_query"], "collections": initial_graph_input["selected_collections"]},
            resp_payload={"error": error_msg},
            meta={"model": GEMINI_MODEL, "used_today": used_today, "error": True}
        )
        
        return jsonify(error_response), 500

def format_conversation_text(query: str, response: str, chunks: List[Dict]) -> str:
    """Format the conversation as plain text."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"Conversation Export - {timestamp}\n\n"
    text += "Question:\n" + query + "\n\n"
    text += "Answer:\n" + response + "\n\n"
    text += "Reference Chunks:\n"
    
    for i, chunk in enumerate(chunks, 1):
        text += f"\nChunk {i}:\n"
        text += f"Collection: {chunk.get('collection', 'N/A')}\n"
        text += f"Text: {chunk.get('text', 'N/A')}\n"
        text += f"Score: {chunk.get('score', 'N/A')}\n"
        
        metadata = chunk.get('metadata', {})
        if metadata:
            text += "Metadata:\n"
            for key, value in metadata.items():
                text += f"  {key}: {value}\n"
        text += "-" * 80 + "\n"
    
    return text

def create_pdf(query: str, response: str, chunks: List[Dict]) -> bytes:
    """Create a PDF document of the conversation."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12
    )
    normal_style = styles['Normal']
    
    # Content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements = []
    
    # Title
    elements.append(Paragraph(f"Conversation Export - {timestamp}", title_style))
    elements.append(Spacer(1, 12))
    
    # Question
    elements.append(Paragraph("Question:", heading_style))
    elements.append(Paragraph(query, normal_style))
    elements.append(Spacer(1, 12))
    
    # Answer
    elements.append(Paragraph("Answer:", heading_style))
    elements.append(Paragraph(response, normal_style))
    elements.append(Spacer(1, 12))
    
    # Reference Chunks
    elements.append(Paragraph("Reference Chunks:", heading_style))
    
    for i, chunk in enumerate(chunks, 1):
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Chunk {i}:", heading_style))
        
        # Create a table for chunk details
        data = [
            ["Collection:", chunk.get('collection', 'N/A')],
            ["Score:", str(chunk.get('score', 'N/A'))],
            ["Text:", chunk.get('text', 'N/A')]
        ]
        
        metadata = chunk.get('metadata', {})
        for key, value in metadata.items():
            data.append([f"{key}:", str(value)])
            
        table = Table(data, colWidths=[1.5*inch, 5*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        elements.append(table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

@app.route('/api/download/<format>', methods=['POST'])
def download_conversation(format):
    # Check authentication
    if not g.user:
        return jsonify({"error": "Authentication required"}), 401
        
    data = request.json
    query = data.get('query', '')
    response = data.get('response', '')
    chunks = data.get('chunks', [])
    
    if format not in ['txt', 'pdf']:
        return jsonify({"error": "Invalid format. Must be 'txt' or 'pdf'"}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == 'txt':
        text_content = format_conversation_text(query, response, chunks)
        buffer = io.BytesIO(text_content.encode('utf-8'))
        filename = f"conversation_{timestamp}.txt"
        mimetype = 'text/plain'
    else:  # pdf
        buffer = io.BytesIO(create_pdf(query, response, chunks))
        filename = f"conversation_{timestamp}.pdf"
        mimetype = 'application/pdf'
    
    # Log the download
    log_interaction(
        user=g.user,
        route=f"/api/download/{format}",
        req_payload={"format": format, "query_length": len(query), "chunks_count": len(chunks)},
        resp_payload={"filename": filename, "format": format},
        meta={"download": True}
    )
    
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )

# Example chat endpoint (mentioned in original integration code)
@app.route("/chat")
@limiter.limit("100 per day", exempt_when=lambda: is_unlimited(getattr(g, "user", None))) if limiter else lambda f: f
@limiter.limit("10 per minute", exempt_when=lambda: is_unlimited(getattr(g, "user", None))) if limiter else lambda f: f
def chat():
    """Simple chat endpoint that echoes the prompt (can be extended)"""
    if not g.user:
        return jsonify({"error": "Authentication required"}), 401
    
    prompt = request.args.get("q", "")
    if not prompt:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    # Simple echo response (this can be extended to use your RAG system)
    answer = {"text": f"Echo: {prompt}"}
    used_today = incr_daily_counter(g.user["sub"]) if redis else 0

    log_interaction(
        user=g.user,
        route="/chat",
        req_payload={"prompt": prompt},
        resp_payload=answer,
        meta={"model": "echo", "used_today": used_today}
    )
    
    return jsonify({"answer": answer, "used_today": used_today})

if __name__ == '__main__':
    if not GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY is not set. Please add it to your .env file.")

    try:
        collections = get_available_collections_internal()
        print(f"Available collections: {collections}")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")

    app.run(debug=True)

#To run:
#/Users/mason/opt/anaconda3/envs/psai/bin/python /Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/code/app2.py
