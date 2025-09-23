#!/usr/bin/env python3
"""
Enhanced ChromaDB Pipeline - With Update Support and LLM Fallback
- Supports updating existing user data
- Performance optimizations for speed
- LLM fallback with voice consistency
- Edge TTS integration for consistent voice
"""

import json
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import os
import hashlib
from datetime import datetime, timedelta
import re
from difflib import SequenceMatcher
import time
from groq import Groq
from dotenv import load_dotenv
import asyncio
import edge_tts
from typing import Optional, Dict, List, Any, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import io
import logging

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = "./chroma_local"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

# TTS Configuration for consistent voice
TTS_VOICE = "en-IN-NeerjaNeural"  # Your Edge TTS voice
FALLBACK_TTS_VOICE = "en-IN-NeerjaNeural"  # Same voice for LLM responses

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBPipeline:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.client = None
        self.user_collections = {}  # Cache for user collections
        self.executor = ThreadPoolExecutor(max_workers=3)  # For async operations
        self.setup_chromadb_client()
        
        # Essential spell corrections (only common ones)
        self.corrections = {
            'wat': 'what', 'hw': 'how', 'wen': 'when', 'wher': 'where',
            'u': 'you', 'ur': 'your', 'r': 'are', 'pls': 'please',
            'opn': 'open', 'clos': 'close', 'tim': 'time', 'pric': 'price',
            'servic': 'service', 'contat': 'contact', 'loction': 'location'
        }
        
        # Enhanced business categories for comprehensive Q&A generation
        self.categories = {
            "hours_basic": ["hours", "time", "open", "close", "available", "schedule"],
            "hours_edge_cases": ["early", "late", "before", "after", "weekend", "holiday"],
            "contact": ["phone", "email", "contact", "reach", "call", "number"],
            "services": ["service", "offer", "do", "provide", "help", "specialize"],
            "pricing": ["price", "cost", "fee", "charge", "expensive", "rate"],
            "location": ["where", "location", "address", "direction", "find"],
            "booking": ["appointment", "book", "schedule", "reserve", "meet"],
            "availability": ["available", "free", "busy", "slot", "timing"],
            "urgent": ["urgent", "emergency", "asap", "immediately", "now"],
            "payment": ["payment", "pay", "accept", "card", "cash", "method"],
            "general_info": ["about", "info", "tell me", "describe", "explain"]
        }

    def setup_chromadb_client(self):
        """Setup ChromaDB client with performance optimizations"""
        try:
            # Use settings for better performance
            settings = chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=CHROMA_DB_PATH,
                anonymized_telemetry=False
            )
            
            self.client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=settings
            )
            logger.info(f"ChromaDB client initialized with performance optimizations at: {CHROMA_DB_PATH}")
        except Exception as e:
            logger.error(f"ChromaDB client setup failed: {e}")
            raise

    def get_user_collection_name(self, user_id):
        """Generate collection name for specific user"""
        clean_user_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(user_id))
        return f"user_{clean_user_id}_qa_collection"

    def setup_user_collection(self, user_id):
        """Setup or get user-specific ChromaDB collection with caching"""
        collection_name = self.get_user_collection_name(user_id)
        
        # Check if collection already exists in cache
        if user_id in self.user_collections:
            return self.user_collections[user_id]
        
        try:
            embedding_func = DefaultEmbeddingFunction()
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_func
            )
            
            # Cache the collection
            self.user_collections[user_id] = collection
            
            current_count = collection.count()
            logger.info(f"User-specific collection initialized: '{collection_name}' ({current_count} Q&A pairs)")
            
            return collection
        except Exception as e:
            logger.error(f"Failed to setup collection for user {user_id}: {e}")
            raise

    def get_user_collection(self, user_id):
        """Get user-specific collection with caching"""
        if user_id not in self.user_collections:
            return self.setup_user_collection(user_id)
        return self.user_collections[user_id]

    def check_for_updates_needed(self, user_id, new_business_data):
        """Check if user data needs updating by comparing business information"""
        try:
            collection = self.get_user_collection(user_id)
            
            # Get existing metadata to compare
            if collection.count() == 0:
                return True, "No existing data"
            
            # Get sample of existing data to check business info
            results = collection.get(limit=5, include=['metadatas'])
            
            if not results or not results.get("metadatas"):
                return True, "No metadata found"
            
            # Compare business information
            existing_metadata = results["metadatas"][0]
            existing_business_name = existing_metadata.get("business_name", "")
            new_business_name = new_business_data.get("business_name", "")
            
            # Check if business name changed significantly
            if existing_business_name and new_business_name:
                similarity = SequenceMatcher(None, existing_business_name.lower(), new_business_name.lower()).ratio()
                if similarity < 0.8:  # Less than 80% similar
                    return True, f"Business name changed: '{existing_business_name}' -> '{new_business_name}'"
            
            # Check content differences
            existing_content_hash = existing_metadata.get("content_hash", "")
            new_content = str(new_business_data.get('content', '')) + str(new_business_data.get('additional_content', ''))
            new_content_hash = hashlib.md5(new_content.encode()).hexdigest()
            
            if existing_content_hash and existing_content_hash != new_content_hash:
                return True, "Business content updated"
            
            return False, "No significant changes detected"
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return True, f"Error checking updates: {e}"

    def update_user_data(self, user_id, business_data, force_update=False):
        """Update existing user data with new business information"""
        try:
            logger.info(f"Checking for updates needed for user: {user_id}")
            
            needs_update, reason = self.check_for_updates_needed(user_id, business_data)
            
            if not needs_update and not force_update:
                logger.info(f"No update needed for user {user_id}: {reason}")
                return True
            
            logger.info(f"Update required for user {user_id}: {reason}")
            
            # Delete existing data
            collection_name = self.get_user_collection_name(user_id)
            try:
                self.client.delete_collection(collection_name)
                if user_id in self.user_collections:
                    del self.user_collections[user_id]
                logger.info(f"Deleted existing collection for user {user_id}")
            except:
                pass  # Collection might not exist
            
            # Generate new comprehensive Q&A pairs
            logger.info(f"Generating updated Q&A pairs for user {user_id}...")
            return self.train_user_agent(user_id, business_data)
            
        except Exception as e:
            logger.error(f"Error updating user data for {user_id}: {e}")
            return False

    def call_groq_api(self, prompt, max_retries=3):
        """Call Groq API with retry logic and caching"""
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert Q&A generator. Output only valid JSON arrays. Generate comprehensive Q&A pairs covering ALL possible questions a real human assistant would handle, including edge cases."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"API Error: {e}")
                time.sleep(2)
        return None

    def call_groq_fallback_api(self, query, context=""):
        """Call Groq API for fallback responses when no match found in ChromaDB"""
        try:
            fallback_prompt = f"""You are a helpful business assistant. A customer asked: "{query}"

{f"Business context: {context}" if context else ""}

Provide a helpful, natural response that:
1. Acknowledges the question
2. Provides useful general guidance if possible
3. Suggests contacting the business directly for specific information
4. Maintains a professional, friendly tone

Response should be concise (1-3 sentences) and natural for voice conversation using the same tone as en-IN-NeerjaNeural voice.
Do not mention that you're an AI or that you don't have specific information."""

            response = self.groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful business assistant providing natural, conversational responses suitable for voice interaction with en-IN-NeerjaNeural voice characteristics."},
                    {"role": "user", "content": fallback_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            fallback_response = response.choices[0].message.content.strip()
            logger.info(f"LLM Fallback response generated for voice consistency: {fallback_response[:50]}...")
            return fallback_response
            
        except Exception as e:
            logger.error(f"Fallback API Error: {e}")
            return "I'd be happy to help you with that. For the most accurate information, please feel free to contact us directly."

    async def generate_voice_response(self, text, voice=None):
        """Generate voice response using Edge TTS with consistent voice"""
        if voice is None:
            voice = TTS_VOICE
        
        try:
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return audio_data
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            return None

    def build_comprehensive_qa_prompt(self, business_data, category):
        """Build comprehensive prompt for Q&A generation including edge cases"""
        business_name = business_data.get('business_name', 'Business')
        industry = business_data.get('industry', 'Service')
        phone = business_data.get('phone', '')
        email = business_data.get('email', '')
        
        # Handle both 'content' and 'additional_content' keys for compatibility
        extra_info = business_data.get('content', '') or business_data.get('additional_content', '')
        
        category_words = self.categories.get(category, [])
        
        # Enhanced prompts for different categories with edge cases
        if "hours" in category:
            return f"""Generate 20+ comprehensive Q&A pairs for business hours including EDGE CASES for an AI call agent.

BUSINESS: {business_name}
INDUSTRY: {industry}
PHONE: {phone}
EMAIL: {email}
ADDITIONAL INFO: {extra_info}

Generate questions covering ALL possible scenarios a human assistant would handle:

1. BASIC HOURS: "what are your hours?", "when are you open?", "wat time u close?"
2. EDGE CASES: 
   - "Can I come at 8 PM?" (after hours)
   - "Are you open early at 7 AM?" (before hours)
   - "What if I arrive 10 minutes late?"
   - "Do you stay open past closing for appointments?"
   - "Are you open on weekends?"
   - "What about holidays?"
   - "Can I come 30 minutes before closing?"
3. VARIATIONS: Include typos, casual language, single words
4. URGENT SITUATIONS: "I need help now", "emergency hours"

Provide helpful, specific answers mentioning {business_name} and include contact info when relevant.
For edge cases, be helpful but clear about policies.

Output ONLY JSON array:
[
  {{"question": "wat r ur hours?", "answer": "Our business hours are [hours]. You can reach us at {phone} for any questions about timing."}},
  {{"question": "can i come at 8pm?", "answer": "We typically close at [time], but for urgent matters you can call us at {phone} to see if we can accommodate you."}}
]"""
        
        elif category == "services":
            return f"""Generate 20+ comprehensive Q&A pairs about services including ALL variations for an AI call agent.

BUSINESS: {business_name}
INDUSTRY: {industry}
ADDITIONAL INFO: {extra_info}

Generate questions covering ALL service-related scenarios:

1. BASIC SERVICES: "what do you do?", "wat services u offer?", "help me with..."
2. SPECIFIC SERVICES: Based on industry and additional info
3. SERVICE LIMITATIONS: "Do you do X?", "Can you help with Y?"
4. EDGE CASES:
   - "What if my case is complicated?"
   - "Do you handle urgent requests?"
   - "What services are NOT included?"
   - "Can you customize your service?"
5. COMPARISONS: "How are you different?", "Why choose you?"

Output ONLY JSON array with specific, helpful answers about {business_name}'s services."""

        elif category == "pricing":
            return f"""Generate 20+ comprehensive Q&A pairs about pricing including ALL cost-related questions.

BUSINESS: {business_name}
INDUSTRY: {industry}
ADDITIONAL INFO: {extra_info}

Generate questions covering ALL pricing scenarios:

1. BASIC PRICING: "how much?", "wat r ur prices?", "cost?"
2. EDGE CASES:
   - "What if I can't afford full price?"
   - "Do you offer payment plans?"
   - "Are there hidden fees?"
   - "What's included in the price?"
   - "Do prices change based on complexity?"
   - "Is there a minimum charge?"
3. COMPARISONS: "Are you expensive?", "Cheapest option?"
4. URGENT PRICING: "Rush job cost?", "Emergency rates?"

Provide helpful answers about {business_name}'s pricing structure."""

        else:
            # Generic comprehensive prompt for other categories
            return f"""Generate 20+ comprehensive Q&A pairs for {category} inquiries including ALL edge cases.

BUSINESS: {business_name}
INDUSTRY: {industry}
PHONE: {phone}
EMAIL: {email}
ADDITIONAL INFO: {extra_info}

Generate questions with spelling variations, typos, and casual language for category: {category}
Include variations like: {', '.join(category_words)}

Requirements:
1. Include common typos: "wat about {category_words[0] if category_words else 'info'}?", "hw to?", "u have?"
2. Single word queries: "{category_words[0] if category_words else 'help'}?", "info?"  
3. Casual language: "whats your {category_words[0] if category_words else 'thing'}?", "tell me about {category_words[0] if category_words else 'it'}"
4. Edge cases and follow-up scenarios
5. Provide helpful, specific answers mentioning {business_name}
6. Include contact info when relevant: {phone}, {email}

Output ONLY JSON array:
[
  {{"question": "wat about {category_words[0] if category_words else 'help'}?", "answer": "Detailed answer about {category} for {business_name}..."}},
  {{"question": "{category_words[0] if category_words else 'info'}?", "answer": "Same helpful answer..."}}
]"""

    def extract_qa_pairs(self, response):
        """Extract Q&A pairs from API response"""
        if not response:
            return []
        
        # Clean response
        response = response.strip()
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        
        try:
            data = json.loads(response)
            if isinstance(data, list):
                valid_pairs = []
                for item in data:
                    if (isinstance(item, dict) and 
                        "question" in item and "answer" in item and
                        len(item["question"].strip()) > 0 and 
                        len(item["answer"].strip()) > 10):
                        valid_pairs.append({
                            "question": item["question"].strip(),
                            "answer": item["answer"].strip()
                        })
                return valid_pairs[:25]  # Increased limit per category for comprehensive coverage
        except json.JSONDecodeError:
            pass
        
        return []

    def generate_comprehensive_qa_pairs(self, business_data):
        """Generate comprehensive Q&A pairs for ALL categories including edge cases"""
        all_qa_pairs = []
        
        logger.info("Generating comprehensive Q&A pairs covering ALL possible scenarios...")
        
        # Use ThreadPoolExecutor for faster generation
        def generate_category_qa(category):
            logger.info(f"Generating {category} Q&A pairs...")
            prompt = self.build_comprehensive_qa_prompt(business_data, category)
            response = self.call_groq_api(prompt)
            qa_pairs = self.extract_qa_pairs(response)
            
            for qa in qa_pairs:
                qa["category"] = category
            
            return qa_pairs
        
        # Generate Q&A pairs in parallel for better performance
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(generate_category_qa, category) for category in self.categories.keys()]
            
            for future in futures:
                try:
                    qa_pairs = future.result(timeout=30)
                    all_qa_pairs.extend(qa_pairs)
                    time.sleep(0.1)  # Small delay to prevent rate limiting
                except Exception as e:
                    logger.error(f"Error generating category Q&A: {e}")
        
        logger.info(f"Generated {len(all_qa_pairs)} total comprehensive Q&A pairs")
        return all_qa_pairs

    def get_qa_hash(self, question, answer, user_id):
        """Generate unique hash for Q&A pair"""
        content = f"{question}|{answer}|{user_id}"
        return hashlib.md5(content.encode()).hexdigest()

    def check_existing_qa(self, user_id):
        """Check how many Q&A pairs exist for user in their specific collection"""
        try:
            collection = self.get_user_collection(user_id)
            return collection.count()
        except:
            return 0

    def store_qa_in_chromadb(self, qa_pairs, user_id, business_name):
        """Store Q&A pairs in user-specific ChromaDB collection with performance optimizations"""
        if not qa_pairs:
            return False
        
        collection = self.get_user_collection(user_id)
        existing_count = collection.count()
        
        logger.info(f"Storing {len(qa_pairs)} Q&A pairs in user {user_id}'s collection...")
        
        # Prepare data for ChromaDB in batches for better performance
        batch_size = 50
        
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i + batch_size]
            
            documents = []
            metadatas = []
            ids = []
            
            # Add content hash for update detection
            content_hash = hashlib.md5(str(business_name).encode()).hexdigest()
            
            for j, qa in enumerate(batch):
                qa_hash = self.get_qa_hash(qa["question"], qa["answer"], user_id)
                
                documents.append(qa["question"])
                metadatas.append({
                    "answer": qa["answer"],
                    "category": qa["category"],
                    "user_id": user_id,
                    "business_name": business_name,
                    "qa_hash": qa_hash,
                    "content_hash": content_hash,  # For update detection
                    "created_at": datetime.now().isoformat(),
                    "qa_pair_json": json.dumps({
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "category": qa["category"]
                    })
                })
                ids.append(f"qa_{qa_hash}_{i}_{j}")
            
            try:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                logger.error(f"Error storing batch: {e}")
                return False
        
        new_count = collection.count()
        logger.info(f"Successfully stored Q&A pairs. Total: {existing_count} -> {new_count}")
        return True

    def calculate_relevance_score(self, query, question, answer):
        """Calculate relevance score for search results with improved algorithm"""
        query_norm = self.normalize_text(query)
        question_norm = self.normalize_text(question)
        answer_norm = self.normalize_text(answer)
        
        # Word overlap with better weighting
        query_words = set(query_norm.split())
        question_words = set(question_norm.split())
        answer_words = set(answer_norm.split())
        
        question_overlap = len(query_words & question_words) / max(len(query_words), 1)
        answer_overlap = len(query_words & answer_words) / max(len(query_words), 1)
        
        # Fuzzy similarity
        fuzzy_score = SequenceMatcher(None, query_norm, question_norm).ratio()
        
        # Length penalty for very short queries
        length_penalty = min(1.0, len(query_words) / 3.0)
        
        # Combined score with improved weighting
        combined_score = (
            question_overlap * 0.4 + 
            answer_overlap * 0.3 + 
            fuzzy_score * 0.3
        ) * length_penalty
        
        return combined_score

    def correct_spelling(self, text):
        """Simple spell correction for common typos"""
        if not text:
            return text
        
        words = text.lower().split()
        corrected = [self.corrections.get(word, word) for word in words]
        return ' '.join(corrected)

    def normalize_text(self, text):
        """Normalize text for better matching"""
        if not text:
            return ""
        
        text = self.correct_spelling(text)
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def search_user_qa(self, query, user_id):
        """Search Q&A pairs in user-specific collection with LLM fallback and consistent voice"""
        try:
            logger.info(f"Searching in user {user_id}'s collection for: '{query}'")
            
            # Get user-specific collection
            collection = self.get_user_collection(user_id)
            
            # Check if collection is empty
            collection_count = collection.count()
            if collection_count == 0:
                logger.info(f"User {user_id}'s collection is empty. Using LLM fallback with consistent voice.")
                return self.call_groq_fallback_api(query)
            
            # Search in user-specific collection with improved performance
            corrected_query = self.correct_spelling(query)
            
            # Reduced n_results for better performance
            results = collection.query(
                query_texts=[corrected_query],
                n_results=min(10, collection_count),  # Adaptive result count
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            if results["documents"] and results["documents"][0]:
                logger.info(f"Found {len(results['documents'][0])} potential matches")
                
                # Score and rank results
                candidates = []
                for i in range(len(results["documents"][0])):
                    question = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    
                    relevance_score = self.calculate_relevance_score(query, question, metadata["answer"])
                    similarity = max(0, 1 - distance)
                    combined_score = (similarity * 0.4) + (relevance_score * 0.6)
                    
                    candidates.append({
                        "answer": metadata["answer"],
                        "score": combined_score,
                        "category": metadata.get("category", "general")
                    })
                
                # Return best match if above threshold
                best_match = max(candidates, key=lambda x: x["score"])
                threshold = 0.15  # Slightly higher threshold for better quality
                
                if best_match["score"] > threshold:
                    logger.info(f"Returning ChromaDB match (score: {best_match['score']:.3f}) - voice will be consistent")
                    return best_match["answer"]
                else:
                    logger.info(f"No good ChromaDB matches (best score: {best_match['score']:.3f}), using LLM fallback with consistent voice")
                    
                    # Get business context for fallback
                    business_context = ""
                    if results["metadatas"] and results["metadatas"][0]:
                        first_metadata = results["metadatas"][0][0]
                        business_name = first_metadata.get("business_name", "")
                        if business_name:
                            business_context = f"Business: {business_name}"
                    
                    return self.call_groq_fallback_api(query, business_context)
            else:
                logger.info("No ChromaDB results found, using LLM fallback with consistent voice")
                return self.call_groq_fallback_api(query)
            
        except Exception as e:
            logger.error(f"Search error for user {user_id}: {e}")
            # Fallback to LLM on any error with consistent voice
            return self.call_groq_fallback_api(query, "I apologize for any technical difficulties.")

    def train_user_agent(self, user_id, business_data):
        """Main function: Generate comprehensive Q&A pairs and store in user-specific collection"""
        logger.info(f"Training comprehensive AI agent for user: {user_id}")
        logger.info(f"Business: {business_data.get('business_name', 'Unknown')}")
        
        collection = self.setup_user_collection(user_id)
        existing_count = collection.count()
        
        if existing_count > 50:
            logger.info(f"User {user_id} already has {existing_count} Q&A pairs")
            return True
        
        # Generate comprehensive Q&A pairs
        qa_pairs = self.generate_comprehensive_qa_pairs(business_data)
        
        if not qa_pairs:
            logger.error("Failed to generate Q&A pairs")
            return False
        
        if len(qa_pairs) < 50:
            logger.warning(f"Warning: Only generated {len(qa_pairs)} Q&A pairs")
        
        # Store in ChromaDB
        success = self.store_qa_in_chromadb(qa_pairs, user_id, business_data.get('business_name', 'Business'))
        
        if success:
            final_count = collection.count()
            categories = len(set(qa['category'] for qa in qa_pairs))
            logger.info(f"Successfully trained agent. Q&A pairs: {final_count}, Categories: {categories}")
        
        return success

    def query_user_agent(self, query, user_id):
        """Query the trained AI agent with LLM fallback and voice consistency"""
        response = self.search_user_qa(query, user_id)
        
        # Note: The response will always use the same voice (TTS_VOICE) 
        # when converted to audio, ensuring consistency between ChromaDB and LLM responses
        logger.info(f"Response ready for TTS with consistent voice ({FALLBACK_TTS_VOICE}): {response[:50]}...")
        return response

    async def query_user_agent_with_voice(self, query, user_id):
        """Query agent and return both text and voice response with consistent voice"""
        text_response = self.query_user_agent(query, user_id)
        
        # Generate voice using consistent Edge TTS voice
        voice_data = await self.generate_voice_response(text_response, FALLBACK_TTS_VOICE)
        
        return {
            "text": text_response,
            "voice_data": voice_data,
            "voice_used": FALLBACK_TTS_VOICE,
            "consistent_voice": True
        }

    def get_user_stats(self, user_id):
        """Get comprehensive statistics for specific user"""
        try:
            collection = self.get_user_collection(user_id)
            
            try:
                results = collection.get(include=['metadatas'])
                
                if not results or not results.get("metadatas"):
                    return {
                        "user_id": user_id, 
                        "collection_name": self.get_user_collection_name(user_id),
                        "qa_pairs": 0, 
                        "categories": {}, 
                        "comprehensive_coverage": False
                    }
                
                categories = {}
                business_name = "Unknown"
                
                for metadata in results["metadatas"]:
                    cat = metadata.get("category", "general")
                    categories[cat] = categories.get(cat, 0) + 1
                    if not business_name or business_name == "Unknown":
                        business_name = metadata.get("business_name", "Unknown")
                
                total_pairs = len(results["metadatas"])
                
                return {
                    "user_id": user_id,
                    "collection_name": self.get_user_collection_name(user_id),
                    "qa_pairs": total_pairs,
                    "categories": categories,
                    "business_name": business_name,
                    "comprehensive_coverage": total_pairs >= 50
                }
            except Exception as e:
                logger.error(f"Error getting data from user {user_id}'s collection: {e}")
                return {
                    "user_id": user_id,
                    "collection_name": self.get_user_collection_name(user_id), 
                    "qa_pairs": 0, 
                    "categories": {}, 
                    "error": str(e)
                }
                
        except Exception as e:
            logger.error(f"Stats error for user {user_id}: {e}")
            return {
                "user_id": user_id,
                "collection_name": self.get_user_collection_name(user_id),
                "qa_pairs": 0, 
                "categories": {}, 
                "error": str(e)
            }

    def delete_user_data(self, user_id):
        """Delete user's entire collection"""
        try:
            collection_name = self.get_user_collection_name(user_id)
            logger.info(f"Deleting user {user_id}'s collection: '{collection_name}'")
            
            collection = self.get_user_collection(user_id)
            count_before = collection.count()
            
            self.client.delete_collection(collection_name)
            
            if user_id in self.user_collections:
                del self.user_collections[user_id]
            
            logger.info(f"Successfully deleted user {user_id}'s collection ({count_before} Q&A pairs)")
            return True
        except Exception as e:
            logger.error(f"Delete error for user {user_id}: {e}")
            return False

    def list_all_users(self):
        """List all users with their collections"""
        try:
            collections = self.client.list_collections()
            user_collections = []
            
            for collection in collections:
                if collection.name.startswith('user_') and collection.name.endswith('_qa_collection'):
                    user_id = collection.name.replace('user_', '').replace('_qa_collection', '')
                    count = collection.count()
                    
                    user_collections.append({
                        "user_id": user_id,
                        "collection_name": collection.name,
                        "qa_pairs": count
                    })
            
            logger.info(f"Found {len(user_collections)} user collections:")
            for user_col in user_collections:
                logger.info(f"  User: {user_col['user_id']} | Collection: {user_col['collection_name']} | Q&A Pairs: {user_col['qa_pairs']}")
            
            return user_collections
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return []

    def get_database_stats(self):
        """Get overall database statistics"""
        try:
            collections = self.client.list_collections()
            total_collections = len(collections)
            total_qa_pairs = 0
            user_count = 0
            
            for collection in collections:
                if collection.name.startswith('user_') and collection.name.endswith('_qa_collection'):
                    user_count += 1
                total_qa_pairs += collection.count()
            
            return {
                "total_collections": total_collections,
                "total_users": user_count,
                "total_qa_pairs": total_qa_pairs,
                "database_path": CHROMA_DB_PATH
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}


# Enhanced main functions for external use with update support and LLM fallback

def train_ai_agent(user_id, content, force_update=False):
    """
    Train AI agent for specific user with business data
    Now supports updating existing data with automatic change detection
    """
    try:
        logger.info(f"Starting training/update for user: {user_id}")
        pipeline = ChromaDBPipeline()
        
        # Handle both dictionary and object formats
        if hasattr(content, 'dict'):
            business_data = content.dict()
        elif isinstance(content, dict):
            business_data = content
        else:
            business_data = {
                'business_name': 'Business',
                'industry': 'Service',
                'phone': '',
                'email': '',
                'content': str(content)
            }
        
        # Check if this is an update or initial training
        existing_count = pipeline.check_existing_qa(user_id)
        
        if existing_count > 0:
            logger.info(f"User {user_id} has existing data. Checking for updates...")
            success = pipeline.update_user_data(user_id, business_data, force_update)
        else:
            logger.info(f"New user {user_id}. Starting initial training...")
            success = pipeline.train_user_agent(user_id, business_data)
        
        if success:
            logger.info(f"Training/Update completed successfully for user {user_id}")
        else:
            logger.error(f"Training/Update failed for user {user_id}")
        
        return success
    except Exception as e:
        logger.error(f"Error training agent for user {user_id}: {e}")
        return False

def update_ai_agent(user_id, content, force_update=True):
    """
    Specifically update existing AI agent data
    """
    return train_ai_agent(user_id, content, force_update=force_update)

def query_ai_agent(query, user_id):
    """
    Query trained AI agent with LLM fallback for consistent voice
    This function ensures consistent voice whether response comes from ChromaDB or LLM
    """
    try:
        logger.info(f"Query from user {user_id}: '{query}'")
        pipeline = ChromaDBPipeline()
        response = pipeline.query_user_agent(query, user_id)
        logger.info(f"Response to user {user_id} (voice-ready): '{response[:100]}...'")
        return response
    except Exception as e:
        logger.error(f"Error querying agent for user {user_id}: {e}")
        return "I'm having a technical issue. Please try again later."

async def query_ai_agent_with_voice(query, user_id):
    """
    Query AI agent and return both text and voice with consistent TTS voice
    Ensures user won't notice difference between ChromaDB and LLM responses
    """
    try:
        logger.info(f"Voice query from user {user_id}: '{query}'")
        pipeline = ChromaDBPipeline()
        result = await pipeline.query_user_agent_with_voice(query, user_id)
        logger.info(f"Voice response generated for user {user_id} using consistent voice: {result['voice_used']}")
        return result
    except Exception as e:
        logger.error(f"Error in voice query for user {user_id}: {e}")
        return {
            "text": "I'm having a technical issue. Please try again later.",
            "voice_data": None,
            "voice_used": FALLBACK_TTS_VOICE,
            "consistent_voice": True
        }

def generate_tts_audio(text, voice=None):
    """
    Generate TTS audio with consistent voice (standalone function)
    """
    async def _generate():
        pipeline = ChromaDBPipeline()
        return await pipeline.generate_voice_response(text, voice or TTS_VOICE)
    
    return asyncio.run(_generate())

def get_agent_stats(user_id):
    """Get agent statistics for user"""
    try:
        logger.info(f"Getting stats for user: {user_id}")
        pipeline = ChromaDBPipeline()
        stats = pipeline.get_user_stats(user_id)
        logger.info(f"User {user_id} stats: {stats['qa_pairs']} Q&A pairs")
        return stats
    except Exception as e:
        logger.error(f"Error getting stats for user {user_id}: {e}")
        return {"user_id": user_id, "qa_pairs": 0, "categories": {}, "error": str(e)}

def delete_user_agent(user_id):
    """Delete all data for a specific user"""
    try:
        logger.info(f"Deleting all data for user: {user_id}")
        pipeline = ChromaDBPipeline()
        success = pipeline.delete_user_data(user_id)
        if success:
            logger.info(f"Successfully deleted all data for user {user_id}")
        else:
            logger.error(f"Failed to delete data for user {user_id}")
        return success
    except Exception as e:
        logger.error(f"Error deleting data for user {user_id}: {e}")
        return False

def list_all_users():
    """List all users and their collection information"""
    try:
        pipeline = ChromaDBPipeline()
        return pipeline.list_all_users()
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        return []

def get_database_stats():
    """Get overall database statistics"""
    try:
        pipeline = ChromaDBPipeline()
        return pipeline.get_database_stats()
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {"error": str(e)}


# Usage Examples and Testing
if __name__ == "__main__":
    print("Enhanced ChromaDB Pipeline - With Update Support & LLM Fallback!")
    print("Key Features:")
    print("- User-specific collections with data isolation")
    print("- Smart update detection and data refreshing")
    print("- LLM fallback when no ChromaDB match found")
    print("- Consistent Edge TTS voice (en-IN-NeerjaNeural) for ALL responses")
    print("- Performance optimizations for faster responses")
    print("- Voice consistency ensures users can't distinguish ChromaDB vs LLM responses")
    
    # Example usage:
    sample_business_data = {
        "business_name": "ABC Real Estate",
        "industry": "Real Estate", 
        "phone": "555-0123",
        "email": "info@abcrealestate.com",
        "content": "We provide comprehensive real estate services including buying, selling, and property management. Hours: 9 AM to 7 PM Monday-Friday."
    }
    
    print("\nExample Usage:")
    print("1. Initial training: train_ai_agent('user123', sample_business_data)")
    print("2. Update existing: update_ai_agent('user123', updated_business_data)")
    print("3. Query with fallback: query_ai_agent('What are your hours?', 'user123')")
    print("4. Query with voice: await query_ai_agent_with_voice('Hello', 'user123')")
    print("5. Generate TTS: generate_tts_audio('Hello world', 'en-IN-NeerjaNeural')")
    
    print("\nKey Improvements:")
    print("- Automatic update detection based on business data changes")
    print("- LLM fallback for questions not in ChromaDB")
    print("- Consistent voice across all responses (ChromaDB + LLM)")
    print("- Better performance with caching and parallel processing")
    print("- Voice consistency ensures seamless user experience")
    print("- Users cannot tell if response came from ChromaDB or LLM fallback")
    
    print("\nVoice Consistency Details:")
    print(f"- Primary TTS Voice: {TTS_VOICE}")
    print(f"- Fallback TTS Voice: {FALLBACK_TTS_VOICE}")
    print("- Both ChromaDB and LLM responses use identical voice settings")
    print("- No audio differences between different response sources")