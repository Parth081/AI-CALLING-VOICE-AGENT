import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pre-defined responses for common queries
RESPONSES = {
    "greeting": [
        "Hello! Welcome to TechFlow Solutions. How can I assist you today?",
        "Hi there! I'm your AI receptionist. What can I help you with?"
    ],
    "services": [
        "We offer comprehensive digital marketing and web development services, including SEO, social media management, and custom website development.",
        "Our services include website design, digital marketing, SEO optimization, and social media management."
    ],
    "hours": [
        "Our business hours are 9 AM to 5 PM, Monday through Friday.",
        "We're open Monday to Friday, from 9 in the morning until 5 in the evening."
    ],
    "contact": [
        "You can reach us at +1 (555) 123-4567 or email us at hello@techflowsolutions.com",
        "Feel free to call us at +1 (555) 123-4567 or send an email to hello@techflowsolutions.com"
    ],
    "location": [
        "We're located in downtown Tech District, but most of our services are provided remotely.",
        "While we operate primarily online, our office is located in the Tech District."
    ],
    "default": [
        "I'm here to help! Would you like to know about our services or schedule an appointment?",
        "I can tell you about our services or help you schedule an appointment. What would you prefer?"
    ]
}

def query_ai_agent(text: str, user_id: str, client_id: str) -> str:
    """
    Query the static AI agent for a response
    """
    logger.info(f"Processing query from user {user_id}: {text}")
    
    # Convert input to lowercase for matching
    text = text.lower()
    
    # Check for different types of queries
    if any(word in text for word in ["hi", "hello", "hey"]):
        return RESPONSES["greeting"][0]
    
    if any(word in text for word in ["service", "offer", "provide", "help"]):
        return RESPONSES["services"][0]
    
    if any(word in text for word in ["hour", "open", "time", "schedule"]):
        return RESPONSES["hours"][0]
    
    if any(word in text for word in ["contact", "phone", "email", "reach"]):
        return RESPONSES["contact"][0]
    
    if any(word in text for word in ["where", "location", "address", "office"]):
        return RESPONSES["location"][0]
    
    # Default response if no specific match
    return RESPONSES["default"][0]

def get_agent_stats(user_id: str) -> Dict[str, Any]:
    """
    Get AI agent statistics
    """
    return {
        "total_queries": 0,  # In static version, we don't track actual counts
        "successful_responses": 0,
        "average_response_time": 0,
        "supported_intents": list(RESPONSES.keys()),
        "response_types": len(RESPONSES),
    }

def train_ai_agent(user_id: str, content: dict) -> bool:
    """
    Train the AI agent with new content
    For static version, this is a mock function
    """
    logger.info(f"Mock training requested for user {user_id}")
    return True

# Example usage
if __name__ == "__main__":
    # Test the static receptionist
    test_queries = [
        "Hello there",
        "What services do you offer?",
        "What are your business hours?",
        "How can I contact you?",
        "Where are you located?",
        "Can you help me?"
    ]
    
    print("Testing Static Receptionist Pipeline:")
    print("-" * 50)
    
    for query in test_queries:
        response = query_ai_agent(query, "test_user", "test_client")
        print(f"\nQ: {query}")
        print(f"A: {response}")
    
    print("\nAgent Stats:", get_agent_stats("test_user"))