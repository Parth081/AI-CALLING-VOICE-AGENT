# booking_module.py - Complete Production-Ready Booking System
import re
import asyncio
import logging
import smtplib
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from googleapiclient.errors import HttpError
import edge_tts
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
# Google Sheets Configuration
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "1GVxiaI7aFvvjVAZ2cgQR5OayUBP_d-sllbKClGQZils")
RANGE_NAME = "sheet1!A:H"  # Extended to include Name, Reason, Email columns
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Email Configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")  # Business owner's email
BUSINESS_NAME = os.getenv("BUSINESS_NAME", "Our Business")

# TTS Configuration (consistent with main system)
TTS_VOICE = "en-IN-NeerjaNeural"

# Business hours configuration
BUSINESS_HOURS = {
    "start": 9,  # 9 AM
    "end": 17,   # 5 PM
    "days": [0, 1, 2, 3, 4]  # Monday to Friday
}
# =======================================================

def is_booking_intent(text: str) -> bool:
    """Enhanced booking intent detection"""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Primary booking keywords
    booking_keywords = [
        "book", "appointment", "schedule", "reserve", 
        "meeting", "slot", "book my", "schedule my",
        "make an appointment", "set up appointment"
    ]
    
    # Time indicators
    time_patterns = [
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
        r"\btomorrow\b", r"\bnext\s+week\b", r"\btoday\b",
        r"\bmonday|tuesday|wednesday|thursday|friday|saturday|sunday\b"
    ]
    
    has_booking_keyword = any(keyword in text_lower for keyword in booking_keywords)
    has_time_pattern = any(re.search(pattern, text_lower) for pattern in time_patterns)
    
    booking_intent = has_booking_keyword or (
        has_time_pattern and any(word in text_lower for word in ["can", "want", "need"])
    )
    
    if booking_intent:
        logger.info(f"Booking intent detected: '{text}'")
    
    return booking_intent


def extract_appointment_details(text: str) -> Dict[str, Any]:
    """
    Extract appointment details from natural language using advanced NLP
    Returns: dict with name, email, reason, and other details
    """
    details = {
        "name": None,
        "email": None,
        "reason": None,
        "phone": None
    }
    
    text_lower = text.lower()
    
    # Extract name patterns (case-sensitive for proper names)
    name_patterns = [
        r"(?:with|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",  # "with John Doe"
        r"(?:appointment\s+for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        r"(?:book|schedule)\s+(?:for\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        r"(?:my\s+name\s+is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text)  # Case-sensitive
        if match:
            potential_name = match.group(1).strip()
            # Validate it's not a common word
            if potential_name not in ["Tomorrow", "Today", "Morning", "Afternoon", "Evening"]:
                details["name"] = potential_name
                break
    
    # Extract email patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    if email_match:
        details["email"] = email_match.group(0)
    
    # Extract phone patterns
    phone_patterns = [
        r'\b(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b',  # 123-456-7890
        r'\b(\(\d{3}\)\s*\d{3}[-.\s]?\d{4})\b',  # (123) 456-7890
        r'\b(\+\d{1,3}\s?\d{10})\b'  # +1 1234567890
    ]
    
    for pattern in phone_patterns:
        phone_match = re.search(pattern, text)
        if phone_match:
            details["phone"] = phone_match.group(1)
            break
    
    # Extract reason/purpose patterns
    reason_patterns = [
        r"(?:for|regarding)\s+([a-z\s]+?)(?:\s+(?:on|at|tomorrow|today|with)|\.|$)",
        r"(?:reason[:\s]+)([a-z\s]+?)(?:\s+(?:on|at)|\.|$)",
        r"(?:about)\s+([a-z\s]+?)(?:\s+(?:on|at)|\.|$)",
        r"(?:to\s+discuss)\s+([a-z\s]+?)(?:\s+(?:on|at)|\.|$)"
    ]
    
    for pattern in reason_patterns:
        match = re.search(pattern, text_lower)
        if match:
            reason = match.group(1).strip()
            # Clean up common stopwords
            reason = re.sub(r'\s+(the|a|an|my|your)$', '', reason)
            if len(reason) > 3:  # Minimum length check
                details["reason"] = reason.title()
                break
    
    logger.info(f"Extracted details: {details}")
    return details


def parse_time_from_text(text: str) -> Optional[datetime]:
    """Enhanced time parsing with multiple patterns"""
    try:
        text_lower = text.lower().strip()
        
        # Pattern 1: Specific times with explicit dates
        # "2025-10-05 at 10:00 AM" or "October 5, 2025 at 10:00 AM"
        explicit_date_patterns = [
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)",
            r"((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4})\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)"
        ]
        
        for pattern in explicit_date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    date_str = match.group(1)
                    hour = int(match.group(2))
                    minute = int(match.group(3) or 0)
                    period = match.group(4)
                    
                    # Parse date
                    if '-' in date_str or '/' in date_str:
                        slot_date = datetime.strptime(date_str.replace('/', '-'), "%Y-%m-%d")
                    else:
                        slot_date = datetime.strptime(date_str, "%B %d, %Y")
                    
                    # Convert to 24-hour format
                    if period == "pm" and hour < 12:
                        hour += 12
                    elif period == "am" and hour == 12:
                        hour = 0
                    
                    return slot_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                except ValueError:
                    continue
        
        # Pattern 2: Standard time patterns (existing logic)
        time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text_lower)
        
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            period = time_match.group(3)
            
            if period == "pm" and hour < 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
            elif not period and hour < 8:
                hour += 12
            
            now = datetime.now()
            slot_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if slot_time <= now:
                slot_time += timedelta(days=1)
            
            return slot_time
        
        # Pattern 3: Day-specific times
        day_patterns = {
            'tomorrow': 1,
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        for day_name, target_weekday in day_patterns.items():
            if day_name in text_lower:
                if day_name == 'tomorrow':
                    base_time = datetime.now() + timedelta(days=1)
                else:
                    now = datetime.now()
                    days_ahead = target_weekday - now.weekday()
                    if days_ahead <= 0:
                        days_ahead += 7
                    base_time = now + timedelta(days=days_ahead)
                
                time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text_lower)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2) or 0)
                    period = time_match.group(3)
                    
                    if period == "pm" and hour < 12:
                        hour += 12
                    elif period == "am" and hour == 12:
                        hour = 0
                    
                    return base_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # Default times for time periods
                if "morning" in text_lower:
                    return base_time.replace(hour=9, minute=0, second=0, microsecond=0)
                elif "afternoon" in text_lower:
                    return base_time.replace(hour=14, minute=0, second=0, microsecond=0)
                elif "evening" in text_lower:
                    return base_time.replace(hour=17, minute=0, second=0, microsecond=0)
                else:
                    return base_time.replace(hour=10, minute=0, second=0, microsecond=0)
        
        return None
        
    except Exception as e:
        logger.error(f"Error parsing time from '{text}': {e}")
        return None


def is_valid_business_hours(slot_time: datetime) -> Tuple[bool, str]:
    """Validate if time is within business hours"""
    if slot_time.weekday() not in BUSINESS_HOURS["days"]:
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        requested_day = weekday_names[slot_time.weekday()]
        business_days = [weekday_names[day] for day in BUSINESS_HOURS["days"]]
        return False, f"We're closed on {requested_day}. We're open {', '.join(business_days)}."
    
    if slot_time.hour < BUSINESS_HOURS["start"]:
        return False, f"We open at {BUSINESS_HOURS['start']}:00 AM. Please choose a later time."
    
    if slot_time.hour >= BUSINESS_HOURS["end"]:
        return False, f"We close at {BUSINESS_HOURS['end']}:00 PM. Please choose an earlier time."
    
    if slot_time <= datetime.now():
        return False, "That time has already passed. Please choose a future time."
    
    if slot_time > datetime.now() + timedelta(days=30):
        return False, "I can only schedule appointments up to 30 days in advance."
    
    return True, "Valid business hours"


async def get_google_sheets_service():
    """Create Google Sheets service with error handling"""
    try:
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            raise FileNotFoundError(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
        
        credentials = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=SCOPES
        )
        service = build("sheets", "v4", credentials=credentials)
        return service
    except Exception as e:
        logger.error(f"Failed to create Google Sheets service: {e}")
        raise Exception(f"Cannot connect to booking system: {str(e)}")


async def check_slot_availability(service, slot_time: datetime) -> Tuple[bool, str]:
    """Check if time slot is available"""
    try:
        sheet = service.spreadsheets()
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: sheet.values().get(
                spreadsheetId=SPREADSHEET_ID,
                range=RANGE_NAME
            ).execute()
        )
        
        rows = result.get("values", [])
        date_str = slot_time.strftime("%Y-%m-%d")
        time_str = slot_time.strftime("%H:%M")
        
        # Check if slot is already taken
        for row in rows[1:]:  # Skip header
            if len(row) >= 2:
                if row[0] == date_str and row[1] == time_str:
                    return False, f"{slot_time.strftime('%I:%M %p on %A, %B %d')} is already booked. Please choose a different time."
        
        return True, f"{slot_time.strftime('%I:%M %p on %A, %B %d')} is available."
        
    except HttpError as e:
        logger.error(f"Google Sheets API error: {e}")
        return False, "Unable to check availability. Please try again."
    except Exception as e:
        logger.error(f"Error checking availability: {e}")
        return False, "Technical issue checking availability."


async def send_booking_email(appointment_data: Dict[str, Any]) -> bool:
    """
    Send email notification with appointment details
    Sends to both user (if email provided) and admin
    """
    try:
        if not SENDER_EMAIL or not SENDER_PASSWORD:
            logger.warning("Email credentials not configured - skipping email")
            return False
        
        slot_time = appointment_data["slot_time"]
        user_name = appointment_data.get("user_name", "Customer")
        user_email = appointment_data.get("user_email")
        reason = appointment_data.get("reason", "General Appointment")
        
        # Prepare email content
        subject = f"Appointment Confirmation - {slot_time.strftime('%B %d, %Y')}"
        
        body = f"""
Dear {user_name},

Your appointment has been successfully confirmed!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPOINTMENT DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Business: {BUSINESS_NAME}
Name: {user_name}
Date: {slot_time.strftime('%A, %B %d, %Y')}
Time: {slot_time.strftime('%I:%M %p')}
Purpose: {reason}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please arrive 5 minutes early for your appointment.

If you need to reschedule or cancel, please contact us as soon as possible.

Thank you for choosing {BUSINESS_NAME}!

Best regards,
{BUSINESS_NAME} Team
        """
        
        # Send to user if email provided
        emails_sent = 0
        recipients = []
        
        if user_email:
            recipients.append(user_email)
        
        if ADMIN_EMAIL:
            recipients.append(ADMIN_EMAIL)
        
        if not recipients:
            logger.warning("No email recipients configured")
            return False
        
        # Send emails
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            
            for recipient in recipients:
                msg = MIMEMultipart()
                msg['From'] = SENDER_EMAIL
                msg['To'] = recipient
                msg['Subject'] = subject
                msg.attach(MIMEText(body, 'plain'))
                
                server.send_message(msg)
                emails_sent += 1
                logger.info(f"Appointment email sent to {recipient}")
        
        return emails_sent > 0
        
    except Exception as e:
        logger.error(f"Failed to send booking email: {e}")
        return False


async def book_appointment(service, slot_time: datetime, user_id: str,
                          user_name: str = None, reason: str = None,
                          user_email: str = None, phone: str = None) -> Tuple[bool, str]:
    """
    Book appointment in Google Sheets with all details
    """
    try:
        date_str = slot_time.strftime("%Y-%m-%d")
        time_str = slot_time.strftime("%H:%M")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare new row with all columns
        new_row = [
            date_str,                              # A: Date
            time_str,                              # B: Time
            user_id,                               # C: User ID
            user_name or "Voice Assistant User",   # D: Name
            "Confirmed",                           # E: Status
            timestamp,                             # F: Created At
            reason or "General Appointment",       # G: Reason
            user_email or ""                       # H: Email
        ]
        
        # Append to Google Sheets
        sheet = service.spreadsheets()
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: sheet.values().append(
                spreadsheetId=SPREADSHEET_ID,
                range=RANGE_NAME,
                valueInputOption="RAW",
                body={"values": [new_row]}
            ).execute()
        )
        
        formatted_time = slot_time.strftime("%I:%M %p on %A, %B %d")
        success_msg = f"Perfect! Your appointment is confirmed for {formatted_time}"
        
        if user_name:
            success_msg += f" for {user_name}"
        
        if reason:
            success_msg += f" regarding {reason}"
        
        success_msg += "."
        
        # Send email notification
        appointment_data = {
            "slot_time": slot_time,
            "user_name": user_name,
            "user_email": user_email,
            "reason": reason,
            "phone": phone
        }
        
        email_sent = await send_booking_email(appointment_data)
        
        if email_sent:
            success_msg += " A confirmation email has been sent."
        
        success_msg += " We'll see you then!"
        
        logger.info(f"Appointment booked: User={user_id}, Time={formatted_time}, Name={user_name}, Reason={reason}")
        return True, success_msg
        
    except HttpError as e:
        logger.error(f"Google Sheets API error: {e}")
        return False, "Unable to complete booking due to a technical issue. Please try again."
    except Exception as e:
        logger.error(f"Error booking appointment: {e}")
        return False, "Sorry, there was a problem booking your appointment. Please try again."


async def handle_booking(transcript: str, user_id: str) -> str:
    """
    Main booking handler with NLP extraction and email notifications
    """
    try:
        logger.info(f"Processing booking: '{transcript}' (user: {user_id})")
        
        # Step 1: Extract appointment details using NLP
        details = extract_appointment_details(transcript)
        
        # Step 2: Parse time from transcript
        slot_time = parse_time_from_text(transcript)
        
        if not slot_time:
            return ("I understand you want to book an appointment, but I need a specific date and time. "
                   "Please say something like: 'Book my appointment for tomorrow at 2 PM' or "
                   "'Schedule an appointment with Dr. Smith on October 5th at 10 AM for consultation'.")
        
        logger.info(f"Parsed time: {slot_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Step 3: Validate business hours
        is_valid, validation_msg = is_valid_business_hours(slot_time)
        if not is_valid:
            return f"{validation_msg} Our business hours are {BUSINESS_HOURS['start']}:00 AM to {BUSINESS_HOURS['end']}:00 PM, Monday through Friday."
        
        # Step 4: Connect to Google Sheets
        try:
            service = await get_google_sheets_service()
        except Exception as service_error:
            logger.error(f"Google Sheets connection failed: {service_error}")
            return "I'm having trouble accessing our booking system. Please try again in a moment or contact us directly."
        
        # Step 5: Check availability
        is_available, availability_msg = await check_slot_availability(service, slot_time)
        if not is_available:
            return availability_msg
        
        # Step 6: Book the appointment
        booking_success, booking_msg = await book_appointment(
            service,
            slot_time,
            user_id,
            user_name=details.get("name"),
            reason=details.get("reason"),
            user_email=details.get("email"),
            phone=details.get("phone")
        )
        
        if booking_success:
            logger.info(f"Booking successful: {user_id} at {slot_time}")
            return booking_msg
        else:
            logger.error(f"Booking failed: {booking_msg}")
            return booking_msg
            
    except Exception as e:
        logger.error(f"Fatal error in booking handler: {e}", exc_info=True)
        return "I apologize, but I encountered an issue while processing your booking. Please try again or contact us directly."


async def generate_booking_tts_response(text: str) -> Optional[bytes]:
    """Generate TTS response with consistent voice"""
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        audio_data = b""
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        return audio_data
    except Exception as e:
        logger.error(f"Booking TTS Error: {e}")
        return None


# ==================== INTEGRATION HELPERS ====================

async def is_booking_query(transcript: str) -> bool:
    """Check if transcript is a booking query"""
    return is_booking_intent(transcript)


async def process_booking_query(transcript: str, user_id: str) -> str:
    """Main integration point for app.py"""
    return await handle_booking(transcript, user_id)


async def get_booking_response_with_voice(transcript: str, user_id: str) -> Dict[str, Any]:
    """
    Get booking response with TTS audio
    Returns dict with text, voice_data, and metadata
    """
    try:
        text_response = await process_booking_query(transcript, user_id)
        voice_data = await generate_booking_tts_response(text_response)
        
        return {
            "text": text_response,
            "voice_data": voice_data,
            "voice_used": TTS_VOICE,
            "booking_processed": True,
            "success": "confirmed" in text_response.lower()
        }
    except Exception as e:
        logger.error(f"Error generating booking response: {e}")
        error_msg = "I'm having trouble with the booking system. Please try again."
        return {
            "text": error_msg,
            "voice_data": None,
            "voice_used": TTS_VOICE,
            "booking_processed": False,
            "success": False
        }


# ==================== TESTING ====================

async def test_booking_system():
    """Test the booking system"""
    print("\n" + "="*60)
    print("TESTING COMPLETE BOOKING SYSTEM")
    print("="*60)
    
    test_queries = [
        "Book my appointment with John Doe on 2025-10-05 at 10:00 AM for consultation",
        "Schedule appointment for tomorrow at 2 PM",
        "Book me at 3:30 PM on Monday for checkup",
        "I need an appointment with Dr. Smith for dental cleaning next Friday at 11 AM",
        "Schedule me for 9 AM tomorrow, my name is Alice Johnson, email alice@example.com"
    ]
    
    test_user = "test_user_123"
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] Query: '{query}'")
        print("-" * 60)
        
        # Test intent detection
        is_booking = is_booking_intent(query)
        print(f"✓ Booking Intent: {is_booking}")
        
        if is_booking:
            # Test detail extraction
            details = extract_appointment_details(query)
            print(f"✓ Extracted Details: {details}")
            
            # Test time parsing
            parsed_time = parse_time_from_text(query)
            if parsed_time:
                print(f"✓ Parsed Time: {parsed_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                print("✗ Could not parse time")
            
            # Test full booking (commented out to avoid actual bookings)
            # result = await handle_booking(query, test_user)
            # print(f"✓ Booking Result: {result}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    print("Enhanced Booking Module - Production Ready")
    print("\nFeatures:")
    print("  ✓ Voice command processing with NLP")
    print("  ✓ Google Sheets integration")
    print("  ✓ Email notifications (user + admin)")
    print("  ✓ Business hours validation")
    print("  ✓ Consistent TTS voice")
    print("  ✓ Complete error handling")
    print("\nConfiguration Required:")
    print("  • .env file with email credentials")
    print("  • service_account.json for Google Sheets")
    print("  • SPREADSHEET_ID in .env or code")
    
    asyncio.run(test_booking_system())