# utils/tts_module.py - FIXED VERSION - Returns bytes instead of file paths
import edge_tts
import asyncio
import tempfile
import os
import uuid
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
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

def get_cached_audio_bytes(cache_key: str) -> Optional[bytes]:
    """Check if audio is cached and return bytes directly"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.wav")
    
    if os.path.exists(cache_file):
        # Check if file is within cache duration
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < CACHE_DURATION:
            try:
                with open(cache_file, 'rb') as f:
                    audio_bytes = f.read()
                print(f"DEBUG: Cache hit for key: {cache_key[:8]}... ({len(audio_bytes)} bytes)")
                return audio_bytes
            except Exception as e:
                print(f"ERROR: Failed to read cached file: {e}")
                # Remove corrupted cache file
                try:
                    os.remove(cache_file)
                except:
                    pass
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
                print(f"DEBUG: Removed old cache file: {os.path.basename(old_file[0])}")
            except:
                pass
                
    except Exception as e:
        print(f"ERROR: Cache cleanup error: {e}")

async def generate_tts_audio_to_file(text: str, output_path: str) -> None:
    """Generate TTS audio to file with error handling"""
    try:
        print(f"DEBUG: Generating TTS audio to: {output_path}")
        communicate = edge_tts.Communicate(
            text, 
            voice=VOICE, 
            rate=RATE, 
            pitch=PITCH
        )
        await communicate.save(output_path)
        
        # Verify file was created and has content
        if not os.path.exists(output_path):
            raise Exception("TTS file not created")
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise Exception("TTS file is empty")
        
        print(f"DEBUG: TTS file generated successfully: {file_size} bytes")
        
    except Exception as e:
        print(f"ERROR: TTS generation failed: {e}")
        # Clean up failed file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        raise e

async def synthesize_speech(text: str, use_cache: bool = True) -> bytes:
    """
    Generate speech from text and return audio bytes directly.
    FIXED: Returns bytes instead of file path to prevent WebSocket audio issues.
    """
    if not text or not text.strip():
        raise ValueError("Empty text provided for TTS")
    
    # Normalize text for better caching
    text = text.strip()
    if len(text) > 500:  # Truncate very long text
        text = text[:500] + "..."
    
    start_time = time.time()
    print(f"DEBUG: Starting TTS for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Check cache first
    if use_cache:
        cache_key = get_cache_key(text, VOICE, RATE, PITCH)
        cached_bytes = get_cached_audio_bytes(cache_key)
        
        if cached_bytes:
            cache_time = time.time() - start_time
            print(f"DEBUG: TTS cache hit completed in {cache_time:.3f}s ({len(cached_bytes)} bytes)")
            return cached_bytes
    
    # Generate new TTS audio
    temp_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.wav")
    
    try:
        # Generate TTS to temporary file
        await generate_tts_audio_to_file(text, temp_path)
        
        # Read audio bytes from file
        with open(temp_path, 'rb') as f:
            audio_bytes = f.read()
        
        if not audio_bytes:
            raise Exception("Generated audio file is empty")
        
        print(f"DEBUG: TTS audio read from file: {len(audio_bytes)} bytes")
        
        # Cache the result if caching is enabled
        if use_cache:
            cache_key = get_cache_key(text, VOICE, RATE, PITCH)
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.wav")
            
            try:
                # Save to cache asynchronously
                async with aiofiles.open(cache_file, 'wb') as cache_f:
                    await cache_f.write(audio_bytes)
                print(f"DEBUG: Cached TTS audio: {cache_key[:8]}... ({len(audio_bytes)} bytes)")
            except Exception as cache_error:
                print(f"WARNING: Cache save error: {cache_error}")
        
        generation_time = time.time() - start_time
        print(f"DEBUG: TTS generation completed in {generation_time:.3f}s")
        
        # Cleanup old cache files periodically
        if use_cache and time.time() % 100 < 1:  # Run cleanup occasionally
            asyncio.create_task(cleanup_old_cache())
        
        return audio_bytes
        
    except Exception as e:
        print(f"ERROR: TTS synthesis failed: {e}")
        raise e
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                print(f"WARNING: Temp file cleanup failed: {cleanup_error}")

# Backward compatibility - old function that returned file path
async def synthesize_speech_to_file(text: str, use_cache: bool = True) -> str:
    """
    Generate speech and return file path (for backward compatibility).
    WARNING: Use synthesize_speech() instead for better WebSocket compatibility.
    """
    print("WARNING: Using deprecated synthesize_speech_to_file(). Use synthesize_speech() for bytes.")
    
    # Get audio bytes
    audio_bytes = await synthesize_speech(text, use_cache)
    
    # Write to temporary file
    output_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.wav")
    
    with open(output_path, 'wb') as f:
        f.write(audio_bytes)
    
    print(f"DEBUG: Audio written to temporary file: {output_path}")
    return output_path

# Warm-up function to pre-initialize Edge TTS
async def warmup_tts():
    """Pre-initialize TTS system with a short phrase"""
    try:
        print("DEBUG: Warming up TTS system...")
        audio_bytes = await synthesize_speech("Hello", use_cache=False)
        if audio_bytes and len(audio_bytes) > 0:
            print(f"DEBUG: TTS system warmed up successfully ({len(audio_bytes)} bytes)")
        else:
            print("WARNING: TTS warmup returned empty audio")
    except Exception as e:
        print(f"ERROR: TTS warmup failed: {e}")

# Cleanup function
def cleanup_tts():
    """Clean up TTS resources"""
    global _EXECUTOR
    if _EXECUTOR:
        _EXECUTOR.shutdown(wait=True)
        print("DEBUG: TTS executor shutdown complete")

# Test function
async def test_tts():
    """Test TTS functionality"""
    print("Testing TTS module...")
    
    test_texts = [
        "Hello, this is a test.",
        "What are your business hours?",
        "Thank you for calling."
    ]
    
    for i, text in enumerate(test_texts, 1):
        try:
            print(f"\nTest {i}: '{text}'")
            
            start_time = time.time()
            audio_bytes = await synthesize_speech(text)
            end_time = time.time()
            
            print(f"  Result: {len(audio_bytes)} bytes in {end_time - start_time:.3f}s")
            
            # Test cache hit
            start_time = time.time()
            cached_bytes = await synthesize_speech(text)
            end_time = time.time()
            
            print(f"  Cache: {len(cached_bytes)} bytes in {end_time - start_time:.3f}s")
            
            if audio_bytes == cached_bytes:
                print("  ✓ Cache working correctly")
            else:
                print("  ✗ Cache mismatch")
                
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    print("Fixed TTS Module - Returns bytes for WebSocket compatibility")
    print("Key fixes:")
    print("- synthesize_speech() now returns bytes directly")
    print("- No more file path issues in WebSocket transmission")
    print("- Better error handling and debugging")
    print("- Cached audio as bytes to prevent file corruption")
    
    # Run test
    asyncio.run(test_tts())