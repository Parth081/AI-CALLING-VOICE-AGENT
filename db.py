# core/db.py - PRODUCTION-READY VERSION
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import threading
import time
import logging
import asyncio
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
import hashlib
from dataclasses import dataclass
from enum import Enum
import json
import os
from collections import OrderedDict

# Configuration
CHROMA_DB_PATH = "./chroma_local"
COLLECTION_NAME = "qa_collection"

# Cache configuration
QUERY_CACHE_SIZE = 1000
CACHE_DURATION = 3600  # 1 hour
SIMILARITY_THRESHOLD = 0.4  # Minimum similarity score
TOP_K_RESULTS = 5  # Number of results to retrieve and evaluate

# Connection pool configuration
MAX_CONCURRENT_QUERIES = 10
QUERY_TIMEOUT = 30  # seconds

# Global variables for connection pooling and caching
_CLIENT = None
_COLLECTION = None
_EMBEDDING_FUNC = None
_LOCK = threading.Lock()
_CONNECTION_SEMAPHORE = None

# Advanced caching system
_QUERY_CACHE = OrderedDict()  # LRU cache
_CACHE_TIMES = {}
_CACHE_STATS = {"hits": 0, "misses": 0, "evictions": 0}

# Logging
logger = logging.getLogger(__name__)

class QueryResult(Enum):
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    LOW_CONFIDENCE = "low_confidence"
    ERROR = "error"

@dataclass
class DatabaseQueryResult:
    """Result container for database queries"""
    answer: str
    confidence: float
    query_type: QueryResult
    response_time_ms: int
    cached: bool = False
    similar_questions: List[str] = None
    metadata: Dict[str, Any] = None
    
    @property
    def is_successful(self) -> bool:
        return self.query_type == QueryResult.SUCCESS

class QueryCache:
    """Advanced query caching with LRU eviction and performance tracking"""
    
    def __init__(self, max_size: int = QUERY_CACHE_SIZE, ttl: int = CACHE_DURATION):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "total_requests": 0}
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry has expired"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl
    
    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
            self.stats["evictions"] += 1
    
    def _make_room(self):
        """Make room in cache using LRU eviction"""
        while len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.timestamps.pop(oldest_key, None)
            self.stats["evictions"] += 1
    
    def get_cache_key(self, query: str, user_id: Optional[str] = None) -> str:
        """Generate normalized cache key including user_id if provided"""
        normalized = query.lower().strip()
        if user_id:
            normalized = f"{user_id}:{normalized}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def get(self, query: str, user_id: Optional[str] = None) -> Optional[DatabaseQueryResult]:
        """Get cached result if available and valid"""
        self.stats["total_requests"] += 1
        cache_key = self.get_cache_key(query, user_id)
        
        # Clean expired entries periodically
        if self.stats["total_requests"] % 100 == 0:
            self._evict_expired()
        
        if cache_key in self.cache and not self._is_expired(cache_key):
            # Move to end (mark as recently used)
            result = self.cache.pop(cache_key)
            self.cache[cache_key] = result
            
            self.stats["hits"] += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            
            # Mark as cached and return
            cached_result = result
            cached_result.cached = True
            return cached_result
        
        self.stats["misses"] += 1
        return None
    
    def put(self, query: str, user_id: Optional[str] = None, result: DatabaseQueryResult = None):
        """Cache query result"""
        cache_key = self.get_cache_key(query, user_id)
        
        # Make room if needed
        self._make_room()
        
        # Store result and timestamp
        self.cache[cache_key] = result
        self.timestamps[cache_key] = time.time()
    
    def clear(self):
        """Clear all cached entries"""
        cleared_count = len(self.cache)
        self.cache.clear()
        self.timestamps.clear()
        logger.info(f"Cache cleared: {cleared_count} entries removed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total = self.stats["total_requests"]
        hit_rate = (self.stats["hits"] / max(1, total)) * 100
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate_percent": round(hit_rate, 1),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "total_requests": total
        }

# Global cache instance
query_cache = QueryCache()

def get_connection_semaphore() -> asyncio.Semaphore:
    """Get semaphore for connection limiting"""
    global _CONNECTION_SEMAPHORE
    if _CONNECTION_SEMAPHORE is None:
        _CONNECTION_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
    return _CONNECTION_SEMAPHORE

def get_db_connection():
    """Get database connection with lazy initialization and thread safety"""
    global _CLIENT, _COLLECTION, _EMBEDDING_FUNC
    
    if _CLIENT is None or _COLLECTION is None:
        with _LOCK:
            if _CLIENT is None or _COLLECTION is None:  # Double-check locking
                logger.info("Initializing ChromaDB connection...")
                start_time = time.perf_counter()
                
                try:
                    _EMBEDDING_FUNC = DefaultEmbeddingFunction()
                    _CLIENT = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                    _COLLECTION = _CLIENT.get_or_create_collection(
                        name=COLLECTION_NAME,
                        embedding_function=_EMBEDDING_FUNC
                    )
                    
                    # Verify collection has data
                    count = _COLLECTION.count()
                    init_time = int((time.perf_counter() - start_time) * 1000)
                    
                    if count == 0:
                        logger.warning("ChromaDB collection is empty - ensure data has been loaded")
                    else:
                        logger.info(f"ChromaDB initialized in {init_time}ms - {count} documents loaded")
                    
                except Exception as e:
                    logger.error(f"ChromaDB initialization failed: {e}")
                    raise
    
    return _COLLECTION

class QueryPreprocessor:
    """Preprocess and enhance user queries for better matching"""
    
    # Common query expansions and synonyms
    QUERY_EXPANSIONS = {
        'price': ['pricing', 'cost', 'charges', 'fee', 'rates', 'money'],
        'hours': ['time', 'schedule', 'open', 'close', 'available', 'timing'],
        'location': ['address', 'where', 'place', 'situated', 'find'],
        'contact': ['phone', 'email', 'reach', 'call', 'connect'],
        'service': ['services', 'offer', 'provide', 'do', 'help'],
        'appointment': ['booking', 'schedule', 'meet', 'visit', 'consultation'],
    }
    
    # Stop words that add noise to queries
    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'that'
    }
    
    @classmethod
    def clean_query(cls, query: str) -> str:
        """Clean and normalize query text"""
        if not query:
            return ""
        
        # Basic cleaning
        query = query.strip().lower()
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove punctuation except question marks and apostrophes
        import re
        query = re.sub(r'[^\w\s\?\']', ' ', query)
        
        return query
    
    @classmethod
    def expand_query(cls, query: str) -> str:
        """Expand query with synonyms and related terms"""
        words = query.split()
        expanded_words = set(words)
        
        for word in words:
            # Add synonyms
            for key, synonyms in cls.QUERY_EXPANSIONS.items():
                if word == key or word in synonyms:
                    expanded_words.update([key] + synonyms[:3])  # Limit expansion
        
        return ' '.join(expanded_words)
    
    @classmethod
    def extract_keywords(cls, query: str) -> List[str]:
        """Extract important keywords from query"""
        words = cls.clean_query(query).split()
        
        # Remove stop words
        keywords = [word for word in words if word not in cls.STOP_WORDS and len(word) > 2]
        
        return keywords
    
    @classmethod
    def preprocess_query(cls, query: str) -> Dict[str, str]:
        """Complete query preprocessing"""
        cleaned = cls.clean_query(query)
        expanded = cls.expand_query(cleaned)
        keywords = cls.extract_keywords(cleaned)
        
        return {
            'original': query,
            'cleaned': cleaned,
            'expanded': expanded,
            'keywords': ' '.join(keywords),
            'keyword_list': keywords
        }

class DatabaseQueryEngine:
    """Advanced query engine with multiple search strategies"""
    
    def __init__(self, collection):
        self.collection = collection
        self.preprocessor = QueryPreprocessor()
    
    async def semantic_search(
        self, 
        query_variants: Dict[str, str], 
        n_results: int = TOP_K_RESULTS,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """Perform semantic search with multiple query variants and optional filtering"""
        
        try:
            # Try different query formulations
            search_queries = [
                query_variants['cleaned'],
                query_variants['expanded'],
                query_variants['keywords']
            ]
            
            best_results = []
            
            for search_query in search_queries:
                if not search_query.strip():
                    continue
                
                try:
                    results = self.collection.query(
                        query_texts=[search_query],
                        n_results=n_results,
                        include=['documents', 'metadatas', 'distances'],
                        where=where  # Apply user_id filter if provided
                    )
                    
                    if results and results.get("documents") and results["documents"][0]:
                        formatted_results = self._format_results(results, search_query)
                        best_results.extend(formatted_results)
                        
                        # If we got good results, we can stop trying other variants
                        if formatted_results and formatted_results[0]['confidence'] > 0.7:
                            break
                            
                except Exception as e:
                    logger.warning(f"Search failed for variant '{search_query}': {e}")
                    continue
            
            # Remove duplicates and sort by confidence
            unique_results = self._deduplicate_results(best_results)
            return sorted(unique_results, key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _format_results(self, raw_results: Dict, query: str) -> List[Dict]:
        """Format raw ChromaDB results into structured format"""
        formatted = []
        
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            if not metadata:
                continue
            
            # Convert distance to confidence (ChromaDB uses cosine distance)
            confidence = max(0.0, 1.0 - (distance / 2.0))
            
            formatted.append({
                'answer': metadata.get('answer', ''),
                'question': doc,
                'confidence': confidence,
                'business': metadata.get('business', ''),
                'category': metadata.get('category', ''),
                'industry': metadata.get('industry', ''),  # Include industry from metadata
                'distance': distance,
                'rank': i
            })
        
        return formatted
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on answer content"""
        seen_answers = set()
        unique_results = []
        
        for result in results:
            answer_hash = hashlib.md5(result['answer'].encode()).hexdigest()
            if answer_hash not in seen_answers:
                seen_answers.add(answer_hash)
                unique_results.append(result)
        
        return unique_results

async def query_database_advanced(
    user_query: str, 
    user_id: Optional[str] = None,
    min_confidence: float = None
) -> DatabaseQueryResult:
    """
    Advanced database query with preprocessing, caching, and comprehensive error handling
    Supports user ID-based filtering for industry data.
    """
    if not user_query or not user_query.strip():
        return DatabaseQueryResult(
            answer="Please provide a valid question.",
            confidence=0.0,
            query_type=QueryResult.NO_RESULTS,
            response_time_ms=0
        )
    
    start_time = time.perf_counter()
    min_confidence = min_confidence or SIMILARITY_THRESHOLD
    
    logger.info(f"Processing query: '{user_query[:100]}{'...' if len(user_query) > 100 else ''}' for user_id: {user_id or 'None'}")
    
    # Check cache first
    cached_result = query_cache.get(user_query, user_id)
    if cached_result:
        query_time = int((time.perf_counter() - start_time) * 1000)
        logger.info(f"Cache hit - query completed in {query_time}ms")
        return cached_result
    
    try:
        # Acquire semaphore for rate limiting
        semaphore = get_connection_semaphore()
        async with semaphore:
            
            # Get database connection
            collection = get_db_connection()
            
            # Preprocess query
            query_variants = QueryPreprocessor.preprocess_query(user_query)
            
            # Initialize query engine
            query_engine = DatabaseQueryEngine(collection)
            
            # Perform semantic search with user_id filter
            where_clause = {"user_id": user_id} if user_id else None
            search_results = await query_engine.semantic_search(
                query_variants, 
                where=where_clause
            )
            
            query_time = int((time.perf_counter() - start_time) * 1000)
            
            if not search_results:
                result = DatabaseQueryResult(
                    answer="No industry data found for this user or query. Could you please rephrase your question or ask about something else?",
                    confidence=0.0,
                    query_type=QueryResult.NO_RESULTS,
                    response_time_ms=query_time,
                    metadata={"preprocessed_query": query_variants, "user_id": user_id}
                )
            else:
                best_match = search_results[0]
                
                if best_match['confidence'] >= min_confidence:
                    # Get similar questions for context
                    similar_questions = [r['question'] for r in search_results[1:4] if r['confidence'] > 0.3]
                    
                    result = DatabaseQueryResult(
                        answer=best_match['answer'],
                        confidence=best_match['confidence'],
                        query_type=QueryResult.SUCCESS,
                        response_time_ms=query_time,
                        similar_questions=similar_questions,
                        metadata={
                            "business": best_match.get('business', ''),
                            "category": best_match.get('category', ''),
                            "industry": best_match.get('industry', ''),  # Include industry from metadata
                            "user_id": user_id,
                            "matched_question": best_match['question'],
                            "preprocessed_query": query_variants,
                            "total_candidates": len(search_results)
                        }
                    )
                    
                    logger.info(f"Query successful - confidence: {best_match['confidence']:.3f}, time: {query_time}ms")
                else:
                    result = DatabaseQueryResult(
                        answer="I'm not confident about my answer to that question. Could you please provide more details or rephrase your question?",
                        confidence=best_match['confidence'],
                        query_type=QueryResult.LOW_CONFIDENCE,
                        response_time_ms=query_time,
                        metadata={
                            "best_match": best_match,
                            "preprocessed_query": query_variants,
                            "user_id": user_id
                        }
                    )
                    
                    logger.warning(f"Low confidence result: {best_match['confidence']:.3f} < {min_confidence}")
            
            # Cache the result
            query_cache.put(user_query, user_id, result)
            
            return result
            
    except asyncio.TimeoutError:
        query_time = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"Database query timed out after {query_time}ms")
        return DatabaseQueryResult(
            answer="I'm experiencing a delay accessing my knowledge base. Please try again.",
            confidence=0.0,
            query_type=QueryResult.ERROR,
            response_time_ms=query_time
        )
        
    except Exception as e:
        query_time = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"Database query failed after {query_time}ms: {e}", exc_info=True)
        return DatabaseQueryResult(
            answer="I'm having trouble accessing my knowledge base right now. Please try again later.",
            confidence=0.0,
            query_type=QueryResult.ERROR,
            response_time_ms=query_time,
            metadata={"error": str(e), "user_id": user_id}
        )

# Legacy compatibility function
def query_database(user_query: str, n_results: int = 3) -> str:
    """
    Legacy synchronous query function for backward compatibility
    """
    try:
        # Run the async function in a new event loop if needed
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        
        if loop is None:
            # No running loop, create new one
            result = asyncio.run(query_database_advanced(user_query))
        else:
            # Running in existing loop, use run_in_executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, query_database_advanced(user_query))
                result = future.result()
        
        return result.answer
        
    except Exception as e:
        logger.error(f"Legacy query function failed: {e}")
        return "Sorry, I'm having trouble accessing my knowledge base."

class DatabaseHealthMonitor:
    """Monitor database health and performance"""
    
    def __init__(self):
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.total_response_time = 0.0
        self.avg_confidence = 0.0
        self.last_health_check = 0
    
    def record_query(self, result: DatabaseQueryResult):
        """Record query metrics"""
        self.total_queries += 1
        self.total_response_time += result.response_time_ms
        
        if result.is_successful:
            self.successful_queries += 1
            # Update rolling average of confidence
            self.avg_confidence = ((self.avg_confidence * (self.successful_queries - 1)) + result.confidence) / self.successful_queries
        else:
            self.failed_queries += 1
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        self.last_health_check = time.time()
        
        try:
            # Test basic connectivity
            collection = get_db_connection()
            
            # Test query performance
            start_time = time.perf_counter()
            test_result = await query_database_advanced("test health check query")
            health_check_time = int((time.perf_counter() - start_time) * 1000)
            
            # Get collection stats
            document_count = collection.count()
            
            # Calculate performance metrics
            success_rate = (self.successful_queries / max(1, self.total_queries)) * 100
            avg_response_time = self.total_response_time / max(1, self.total_queries)
            
            return {
                "status": "healthy" if document_count > 0 else "degraded",
                "database": {
                    "document_count": document_count,
                    "collection_name": COLLECTION_NAME,
                    "health_check_time_ms": health_check_time
                },
                "performance": {
                    "total_queries": self.total_queries,
                    "success_rate_percent": round(success_rate, 1),
                    "avg_response_time_ms": round(avg_response_time, 0),
                    "avg_confidence": round(self.avg_confidence, 3)
                },
                "cache": query_cache.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

# Global health monitor
health_monitor = DatabaseHealthMonitor()

def get_database_stats() -> Dict[str, Any]:
    """Get current database statistics"""
    try:
        collection = get_db_connection()
        count = collection.count()
        
        return {
            "total_documents": count,
            "collection_name": COLLECTION_NAME,
            "status": "healthy" if count > 0 else "empty",
            "cache": query_cache.get_stats(),
            "performance": {
                "total_queries": health_monitor.total_queries,
                "successful_queries": health_monitor.successful_queries,
                "failed_queries": health_monitor.failed_queries,
                "success_rate": (health_monitor.successful_queries / max(1, health_monitor.total_queries)) * 100
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

def clear_cache():
    """Clear query cache and reset stats"""
    query_cache.clear()
    logger.info("Database cache cleared")

async def warmup_database():
    """Pre-warm database connection and perform health check"""
    try:
        logger.info("Warming up database connection...")
        start_time = time.perf_counter()
        
        # Initialize connection
        collection = get_db_connection()
        
        # Perform test query
        test_result = await query_database_advanced("hello")
        
        # Run health check
        health_result = await health_monitor.health_check()
        
        warmup_time = int((time.perf_counter() - start_time) * 1000)
        
        logger.info(f"Database warmed up in {warmup_time}ms - Status: {health_result['status']}")
        
        return health_result['status'] in ['healthy', 'degraded']
        
    except Exception as e:
        logger.error(f"Database warmup failed: {e}")
        return False

def warmup_database_sync():
    """Synchronous wrapper for database warmup"""
    try:
        return asyncio.run(warmup_database())
    except Exception as e:
        logger.error(f"Sync database warmup failed: {e}")
        return False

# Cleanup function
def cleanup_db():
    """Clean up database resources"""
    global _CLIENT, _COLLECTION, _CONNECTION_SEMAPHORE
    
    try:
        if _CLIENT:
            # ChromaDB handles cleanup automatically
            _CLIENT = None
            _COLLECTION = None
        
        if _CONNECTION_SEMAPHORE:
            _CONNECTION_SEMAPHORE = None
        
        # Clear cache
        clear_cache()
        
        logger.info("Database resources cleaned up")
        
    except Exception as e:
        logger.error(f"Database cleanup error: {e}")

# Query performance decorator
def monitor_query_performance(func):
    """Decorator to monitor query performance"""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            if isinstance(result, DatabaseQueryResult):
                health_monitor.record_query(result)
            return result
        except Exception as e:
            # Record failed query
            failed_result = DatabaseQueryResult(
                answer="Query failed",
                confidence=0.0,
                query_type=QueryResult.ERROR,
                response_time_ms=int((time.perf_counter() - start_time) * 1000)
            )
            health_monitor.record_query(failed_result)
            raise
    return wrapper

# Export functions for external use
__all__ = [
    'query_database',
    'query_database_advanced', 
    'get_database_stats',
    'clear_cache',
    'warmup_database',
    'warmup_database_sync',
    'cleanup_db',
    'health_monitor',
    'DatabaseQueryResult',
    'QueryResult'
]