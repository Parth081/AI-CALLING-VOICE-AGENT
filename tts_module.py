# utils/tts_module.py - OPTIMIZED VERSION
import edge_tts
import asyncio
import tempfile
import os
import uuid
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional  # Added missing import
import aiofiles

# Configuration
VOICE = os.getenv("EDGE_TTS_VOICE", "en-IN-NeerjaNeural")
RATE = os.getenv("EDGE_TTS_RATE", "+0%") 
PITCH = os.getenv("EDGE_TTS_PITCH", "+0Hz")

# Cache configuration
CACHE_DIR = os.path.join(tempfile.gettempdir(), "tts_cache")
MAX_CACHE_SIZE = 100  # Maximum cached files
CACHE_DURATION = 3600  # 1 hour cache duration

# Thread pool for I/O operations
_EXECUTOR = ThreadPoolExecutor(max_workers=3)

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(text: str, voice: str, rate: str, pitch: str) -> str:
    """Generate cache key for TTS request"""
    cache_input = f"{text}_{voice}_{rate}_{pitch}"
    return hashlib.md5(cache_input.encode()).hexdigest()

def get_cached_audio(cache_key: str) -> Optional[str]:
    """Check if audio is cached and still valid"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.wav")
    
    if os.path.exists(cache_file):
        # Check if file is within cache duration
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < CACHE_DURATION:
            print(f"Cache hit for key: {cache_key[:8]}...")
            return cache_file
        else:
            # Remove expired cache
            try:
                os.remove(cache_file)
            except:
                pass
    
    return None

async def cleanup_old_cache():
    """Remove old cache files to maintain cache size"""
    try:
        cache_files = []
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith('.wav'):
                filepath = os.path.join(CACHE_DIR, filename)
                cache_files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Remove oldest files if cache size exceeded
        while len(cache_files) > MAX_CACHE_SIZE:
            old_file = cache_files.pop(0)
            try:
                os.remove(old_file[0])
                print(f"Removed old cache file: {os.path.basename(old_file[0])}")
            except:
                pass
                
    except Exception as e:
        print(f"Cache cleanup error: {e}")

async def generate_tts_audio(text: str, output_path: str) -> None:
    """Generate TTS audio with error handling"""
    try:
        communicate = edge_tts.Communicate(
            text, 
            voice=VOICE, 
            rate=RATE, 
            pitch=PITCH
        )
        await communicate.save(output_path)
        
        # Verify file was created and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception("TTS generation failed - empty or missing file")
            
    except Exception as e:
        print(f"TTS generation error: {e}")
        raise e

async def synthesize_speech(text: str, use_cache: bool = True) -> str:
    """
    Generate speech from text with caching and optimizations.
    Returns the path to the generated wav file.
    """
    if not text or not text.strip():
        raise ValueError("Empty text provided for TTS")
    
    # Normalize text for better caching
    text = text.strip()
    if len(text) > 500:  # Truncate very long text
        text = text[:500] + "..."
    
    start_time = time.time()
    print(f"Starting TTS for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Check cache first
    if use_cache:
        cache_key = get_cache_key(text, VOICE, RATE, PITCH)
        cached_file = get_cached_audio(cache_key)
        
        if cached_file:
            # Create a copy of cached file for return
            output_path = os.path.join(
                tempfile.gettempdir(), 
                f"tts_{uuid.uuid4().hex}.wav"
            )
            
            # Copy cached file asynchronously
            async with aiofiles.open(cached_file, 'rb') as src:
                async with aiofiles.open(output_path, 'wb') as dst:
                    await dst.write(await src.read())
            
            print(f"TTS cache hit completed in {time.time() - start_time:.2f}s")
            return output_path
    
    # Generate new TTS audio
    output_path = os.path.join(
        tempfile.gettempdir(), 
        f"tts_{uuid.uuid4().hex}.wav"
    )
    
    try:
        # Generate TTS
        await generate_tts_audio(text, output_path)
        
        # Cache the result if caching is enabled
        if use_cache:
            cache_key = get_cache_key(text, VOICE, RATE, PITCH)
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.wav")
            
            try:
                # Copy to cache asynchronously
                async with aiofiles.open(output_path, 'rb') as src:
                    async with aiofiles.open(cache_file, 'wb') as dst:
                        await dst.write(await src.read())
                print(f"Cached TTS audio: {cache_key[:8]}...")
            except Exception as cache_error:
                print(f"Cache save error: {cache_error}")
        
        generation_time = time.time() - start_time
        print(f"TTS generation completed in {generation_time:.2f}s")
        
        # Cleanup old cache files periodically
        if use_cache and time.time() % 100 < 1:  # Run cleanup occasionally
            asyncio.create_task(cleanup_old_cache())
        
        return output_path
        
    except Exception as e:
        # Cleanup failed file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise e

# Warm-up function to pre-initialize Edge TTS
async def warmup_tts():
    """Pre-initialize TTS system with a short phrase"""
    try:
        print("Warming up TTS system...")
        temp_file = await synthesize_speech("Hello", use_cache=False)
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print("TTS system warmed up")
    except Exception as e:
        print(f"TTS warmup failed: {e}")

# Cleanup function
def cleanup_tts():
    """Clean up TTS resources"""
    global _EXECUTOR
    if _EXECUTOR:
        _EXECUTOR.shutdown(wait=True)