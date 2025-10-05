#!/usr/bin/env python3
"""
Production-Ready Q&A Pipeline - SIMPLIFIED & WORKING
- Generates 400+ Q&A pairs reliably
- Fast and optimized
- No complex prompts - direct generation
"""

import json
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import os
import hashlib
from datetime import datetime
import re
from difflib import SequenceMatcher
import time
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
import edge_tts
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from functools import lru_cache
import threading

load_dotenv()

# Configuration
CHROMA_DB_PATH = "./chroma_local"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
OPENAI_MODEL = "gpt-4o-mini"
TTS_VOICE = "en-IN-NeerjaNeural"

# Performance settings
MAX_WORKERS = 8
BATCH_SIZE = 100
CACHE_SIZE = 1000

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartQAPipeline:
    """Simplified, reliable Q&A generation"""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY required")
        
        self.openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            max_retries=2,
            timeout=45.0
        )
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.user_collections = {}
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._lock = threading.Lock()
        
        logger.info("Smart Q&A Pipeline initialized")
        
        # ChromaDB
        try:
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            logger.info(f"ChromaDB ready at {CHROMA_DB_PATH}")
        except Exception as e:
            logger.error(f"ChromaDB failed: {e}")
            raise
        
        # Spell corrections
        self.corrections = {
            'wat': 'what', 'wht': 'what', 'hw': 'how', 'wen': 'when', 'wher': 'where',
            'u': 'you', 'ur': 'your', 'r': 'are', 'pls': 'please', 'plz': 'please',
            'opn': 'open', 'clos': 'close', 'tim': 'time', 'pric': 'price',
            'gud': 'good', 'grt': 'great', 'thx': 'thanks', 'nw': 'now',
            'tmrw': 'tomorrow', 'tdy': 'today', 'wrk': 'work', 'wanna': 'want to',
            'gonna': 'going to', 'gotta': 'got to', 'lemme': 'let me'
        }
        
        # Query cache
        self._query_cache = {}
        self._cache_lock = threading.Lock()

    def get_collection_name(self, user_id: str) -> str:
        clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(user_id))
        return f"user_{clean_id}_qa"

    def setup_collection(self, user_id: str):
        collection_name = self.get_collection_name(user_id)
        
        with self._lock:
            if user_id in self.user_collections:
                return self.user_collections[user_id]
        
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=DefaultEmbeddingFunction()
            )
            
            with self._lock:
                self.user_collections[user_id] = collection
            
            logger.info(f"Collection ready: {collection_name} ({collection.count()} pairs)")
            return collection
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

    def get_collection(self, user_id: str):
        with self._lock:
            if user_id in self.user_collections:
                return self.user_collections[user_id]
        return self.setup_collection(user_id)

    def check_existing(self, user_id: str) -> int:
        try:
            collection = self.get_collection(user_id)
            return collection.count()
        except:
            return 0

    def call_openai_simple(self, prompt: str, max_tokens: int = 4000) -> Optional[str]:
        """Simplified OpenAI call - more reliable"""
        try:
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate realistic customer questions and helpful answers. Output only valid JSON arrays."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None

    def call_groq(self, prompt: str) -> Optional[str]:
        """Groq fallback"""
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return None

    def generate_batch(self, batch_num: int, business_info: str, business_name: str) -> List[Dict]:
        """Generate one batch of 50 Q&A pairs"""
        
        prompt = f"""Generate 50 real customer questions and answers about this business.

BUSINESS INFO:
{business_info}

RULES:
1. Mix question types: ultra-short ("open?"), casual ("wat time u open"), natural ("Do you have X?")
2. Cover: location, hours, services, prices, contact, booking, payment, features, policies
3. Use exact info from business data above
4. Include typos and casual language
5. Each answer must mention "{business_name}" and be specific

OUTPUT: Valid JSON only, no markdown:
[
  {{"question": "open?", "answer": "Yes, {business_name} is open [exact hours]."}},
  {{"question": "wer r u", "answer": "{business_name} is located at [exact address]."}},
  {{"question": "how much", "answer": "At {business_name}, [specific prices]."}}
]

Generate exactly 50 Q&A pairs."""

        response = self.call_openai_simple(prompt, max_tokens=4000)
        
        if not response:
            logger.warning(f"Batch {batch_num} failed")
            return []
        
        pairs = self.extract_qa_pairs(response)
        logger.info(f"Batch {batch_num}: {len(pairs)} pairs")
        return pairs

    def generate_all_batches(self, business_data: Dict) -> List[Dict]:
        """Generate 8 batches of 50 = 400 Q&A pairs in parallel"""
        
        business_name = business_data.get('business_name', 'Business')
        industry = business_data.get('industry', 'Service')
        phone = business_data.get('phone', '')
        email = business_data.get('email', '')
        content = business_data.get('content', '') or business_data.get('additional_content', '')
        
        # Compact business info
        business_info = f"""Business: {business_name}
Industry: {industry}
Phone: {phone}
Email: {email}

Details:
{content[:2000]}"""  # Limit to 2000 chars to avoid timeouts
        
        logger.info("Generating 400+ Q&A pairs in parallel...")
        
        all_pairs = []
        futures = []
        
        # Launch 8 parallel batches
        for i in range(8):
            future = self.executor.submit(
                self.generate_batch,
                i + 1,
                business_info,
                business_name
            )
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                pairs = future.result(timeout=60)
                all_pairs.extend(pairs)
            except Exception as e:
                logger.error(f"Batch failed: {e}")
        
        logger.info(f"Generated {len(all_pairs)} total pairs")
        
        # Deduplicate
        unique = self.smart_deduplicate(all_pairs)
        logger.info(f"After deduplication: {len(unique)} pairs")
        
        return unique

    def smart_deduplicate(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Remove duplicates"""
        seen = {}
        unique = []
        
        for qa in qa_pairs:
            if not qa.get('question') or not qa.get('answer'):
                continue
            
            q = self.normalize_text(qa['question'])
            if len(q.split()) < 1:
                continue
            
            # Create signature
            words = sorted(set(q.split()))[:5]
            sig = ' '.join(words)
            
            if sig not in seen:
                seen[sig] = qa
                unique.append(qa)
            elif len(qa['answer']) > len(seen[sig]['answer']) * 1.2:
                # Replace with better answer
                idx = unique.index(seen[sig])
                unique[idx] = qa
                seen[sig] = qa
        
        return unique

    def extract_qa_pairs(self, response: Optional[str]) -> List[Dict]:
        """Extract Q&A from JSON"""
        if not response:
            return []
        
        response = response.strip()
        
        # Remove markdown
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        try:
            data = json.loads(response)
            if not isinstance(data, list):
                return []
            
            valid = []
            for item in data:
                if (isinstance(item, dict) and 
                    'question' in item and 
                    'answer' in item and
                    len(item['question'].strip()) > 0 and 
                    len(item['answer'].strip()) > 10):
                    valid.append({
                        'question': item['question'].strip(),
                        'answer': item['answer'].strip()
                    })
            
            return valid
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return []

    def store_qa(self, qa_pairs: List[Dict], user_id: str, business_name: str) -> bool:
        """Store in ChromaDB"""
        if not qa_pairs:
            return False
        
        logger.info(f"Storing {len(qa_pairs)} pairs...")
        
        collection = self.get_collection(user_id)
        
        for i in range(0, len(qa_pairs), BATCH_SIZE):
            batch = qa_pairs[i:i + BATCH_SIZE]
            
            documents = []
            metadatas = []
            ids = []
            
            for j, qa in enumerate(batch):
                qa_hash = hashlib.md5(
                    f"{qa['question']}|{qa['answer']}|{user_id}".encode()
                ).hexdigest()[:16]
                
                documents.append(qa['question'])
                metadatas.append({
                    'answer': qa['answer'],
                    'user_id': user_id,
                    'business_name': business_name,
                    'created_at': datetime.now().isoformat()
                })
                ids.append(f"qa_{qa_hash}_{i}_{j}")
            
            try:
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
            except Exception as e:
                logger.error(f"Storage failed: {e}")
                return False
        
        logger.info(f"Stored successfully. Total: {collection.count()}")
        return True

    @lru_cache(maxsize=CACHE_SIZE)
    def correct_spelling(self, text: str) -> str:
        if not text:
            return text
        words = text.lower().split()
        return ' '.join([self.corrections.get(w, w) for w in words])

    @lru_cache(maxsize=CACHE_SIZE)
    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = self.correct_spelling(text)
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return re.sub(r'\s+', ' ', text).strip()

    def calculate_relevance(self, query: str, question: str, answer: str) -> float:
        q_norm = self.normalize_text(query)
        ques_norm = self.normalize_text(question)
        ans_norm = self.normalize_text(answer)
        
        q_words = set(q_norm.split())
        ques_words = set(ques_norm.split())
        ans_words = set(ans_norm.split())
        
        if not q_words:
            return 0.0
        
        ques_overlap = len(q_words & ques_words) / len(q_words)
        ans_overlap = len(q_words & ans_words) / len(q_words)
        fuzzy = SequenceMatcher(None, q_norm, ques_norm).ratio()
        
        return (ques_overlap * 0.4) + (ans_overlap * 0.3) + (fuzzy * 0.3)

    def groq_fallback(self, query: str, context: str = "") -> str:
        try:
            prompt = f"""Answer this customer question briefly: "{query}"
{f"Business: {context}" if context else ""}"""
            response = self.call_groq(prompt)
            return response if response else "Please contact us directly for details."
        except:
            return "For accurate information, please contact us directly."

    def search_qa(self, query: str, user_id: str) -> str:
        """Search with caching"""
        
        # Check cache
        cache_key = f"{user_id}:{query.lower()}"
        with self._cache_lock:
            if cache_key in self._query_cache:
                return self._query_cache[cache_key]
        
        try:
            collection = self.get_collection(user_id)
            count = collection.count()
            
            if count == 0:
                return self.groq_fallback(query)
            
            corrected = self.correct_spelling(query)
            
            results = collection.query(
                query_texts=[corrected],
                n_results=min(5, count),
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return self.groq_fallback(query)
            
            candidates = []
            for i in range(len(results['documents'][0])):
                question = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if results['distances'] else 1.0
                
                relevance = self.calculate_relevance(query, question, metadata['answer'])
                similarity = max(0, 1 - distance)
                score = (similarity * 0.4) + (relevance * 0.6)
                
                candidates.append({
                    'answer': metadata['answer'],
                    'score': score,
                    'business': metadata.get('business_name', '')
                })
            
            best = max(candidates, key=lambda x: x['score'])
            
            if best['score'] > 0.15:
                answer = best['answer']
                
                # Cache it
                with self._cache_lock:
                    if len(self._query_cache) > CACHE_SIZE:
                        self._query_cache.pop(next(iter(self._query_cache)))
                    self._query_cache[cache_key] = answer
                
                return answer
            else:
                return self.groq_fallback(query, best['business'])
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self.groq_fallback(query)

    async def generate_voice(self, text: str, voice: str = None) -> Optional[bytes]:
        try:
            communicate = edge_tts.Communicate(text, voice or TTS_VOICE)
            audio = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
            return audio if audio else None
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    def train(self, user_id: str, business_data: Dict) -> bool:
        """Train with parallel batch generation"""
        logger.info(f"Training user: {user_id}")
        logger.info(f"Business: {business_data.get('business_name')}")
        
        collection = self.setup_collection(user_id)
        existing = collection.count()
        
        if existing > 300:
            logger.info(f"Already trained ({existing} pairs)")
            return True
        
        start = time.time()
        
        # Generate in parallel batches
        qa_pairs = self.generate_all_batches(business_data)
        
        duration = time.time() - start
        
        if not qa_pairs:
            logger.error("Generation failed - no pairs created")
            return False
        
        logger.info(f"Generated {len(qa_pairs)} pairs in {duration:.1f}s")
        
        if len(qa_pairs) < 200:
            logger.warning(f"Only {len(qa_pairs)} pairs (expected 300+)")
        
        success = self.store_qa(
            qa_pairs,
            user_id,
            business_data.get('business_name', 'Business')
        )
        
        if success:
            logger.info(f"Training complete: {collection.count()} total pairs")
        
        return success

    def query(self, query: str, user_id: str) -> str:
        return self.search_qa(query, user_id)

    async def query_with_voice(self, query: str, user_id: str) -> Dict:
        text = self.query(query, user_id)
        voice = await self.generate_voice(text)
        return {'text': text, 'voice_data': voice, 'voice_used': TTS_VOICE}

    def get_stats(self, user_id: str) -> Dict:
        try:
            collection = self.get_collection(user_id)
            count = collection.count()
            
            if count == 0:
                return {'user_id': user_id, 'qa_pairs': 0}
            
            results = collection.get(include=['metadatas'], limit=1)
            business_name = results['metadatas'][0].get('business_name', 'Unknown') if results['metadatas'] else 'Unknown'
            
            return {
                'user_id': user_id,
                'qa_pairs': count,
                'business_name': business_name,
                'cache_size': len(self._query_cache)
            }
        except:
            return {'user_id': user_id, 'qa_pairs': 0, 'error': 'Failed'}

    def delete_user(self, user_id: str) -> bool:
        try:
            name = self.get_collection_name(user_id)
            self.client.delete_collection(name)
            
            with self._lock:
                if user_id in self.user_collections:
                    del self.user_collections[user_id]
            
            with self._cache_lock:
                keys_to_remove = [k for k in self._query_cache.keys() if k.startswith(f"{user_id}:")]
                for key in keys_to_remove:
                    del self._query_cache[key]
            
            return True
        except:
            return False

    def list_users(self) -> List[Dict]:
        try:
            collections = self.client.list_collections()
            return [
                {
                    'user_id': col.name.replace('user_', '').replace('_qa', ''),
                    'qa_pairs': col.count()
                }
                for col in collections if col.name.startswith('user_')
            ]
        except:
            return []

    def get_db_stats(self) -> Dict:
        try:
            collections = self.client.list_collections()
            user_collections = [c for c in collections if c.name.startswith('user_')]
            
            return {
                'total_users': len(user_collections),
                'total_qa_pairs': sum(c.count() for c in user_collections),
                'database_path': CHROMA_DB_PATH,
                'model': OPENAI_MODEL,
                'cache_size': len(self._query_cache)
            }
        except:
            return {'error': 'Unavailable'}


# External API
_pipeline = None
_pipeline_lock = threading.Lock()


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = SmartQAPipeline()
    return _pipeline


def train_ai_agent(user_id: str, content, force_update: bool = False) -> bool:
    try:
        pipeline = _get_pipeline()
        
        if hasattr(content, 'dict'):
            data = content.dict()
        elif isinstance(content, dict):
            data = content
        else:
            data = {
                'business_name': 'Business',
                'industry': 'Service',
                'phone': '',
                'email': '',
                'content': str(content)
            }
        
        existing = pipeline.check_existing(user_id)
        
        if existing > 300 and not force_update:
            logger.info(f"Already trained ({existing} pairs)")
            return True
        
        if existing > 0 and force_update:
            try:
                pipeline.delete_user(user_id)
            except:
                pass
        
        return pipeline.train(user_id, data)
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def query_ai_agent(query: str, user_id: str) -> str:
    try:
        pipeline = _get_pipeline()
        return pipeline.query(query, user_id)
    except:
        return "Technical issue. Please try again."


async def query_ai_agent_with_voice(query: str, user_id: str) -> Dict:
    try:
        pipeline = _get_pipeline()
        return await pipeline.query_with_voice(query, user_id)
    except:
        return {'text': "Technical issue.", 'voice_data': None}


def generate_tts_audio(text: str, voice: str = None) -> Optional[bytes]:
    async def _gen():
        pipeline = _get_pipeline()
        return await pipeline.generate_voice(text, voice or TTS_VOICE)
    return asyncio.run(_gen())


def get_agent_stats(user_id: str) -> Dict:
    try:
        pipeline = _get_pipeline()
        return pipeline.get_stats(user_id)
    except:
        return {'user_id': user_id, 'qa_pairs': 0, 'error': 'Failed'}


def delete_user_agent(user_id: str) -> bool:
    try:
        pipeline = _get_pipeline()
        return pipeline.delete_user(user_id)
    except:
        return False


def list_all_users() -> List[Dict]:
    try:
        pipeline = _get_pipeline()
        return pipeline.list_users()
    except:
        return []


def get_database_stats() -> Dict:
    try:
        pipeline = _get_pipeline()
        return pipeline.get_db_stats()
    except:
        return {'error': 'Unavailable'}


if __name__ == "__main__":
    print("=" * 80)
    print("SIMPLIFIED Q&A PIPELINE - PRODUCTION READY")
    print("=" * 80)
    print("\nFEATURES:")
    print("  - Generates 400+ Q&A pairs reliably")
    print("  - 8 parallel batches of 50 questions each")
    print("  - Simplified prompts - no timeouts")
    print("  - Fast query with caching")
    print("  - Universal business support")
    print(f"\nMODEL: {OPENAI_MODEL}")
    print(f"DATABASE: {CHROMA_DB_PATH}")
    print("=" * 80)