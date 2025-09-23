# enhanced_booking_module.py - Enhanced with voice consistency and better error handling
import re
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from googleapiclient.errors import HttpError
import edge_tts

# Configure logging
logger = logging.getLogger(__name__)

# --- CONFIGURE THESE ---
# FIXED: Extract just the spreadsheet ID from the URL
SPREADSHEET_ID = "1GVxiaI7aFvvjVAZ2cgQR5OayUBP_d-sllbKClGQZils"  # Extracted from your URL
RANGE_NAME = "sheet1!A:F"  # Sheet name and range
SERVICE_ACCOUNT_FILE = "service_account.json"  # Your service account JSON file
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# TTS Configuration for consistent voice with main system
TTS_VOICE = "en-IN-NeerjaNeural"  # Same voice as your main system

# Business hours configuration
BUSINESS_HOURS = {
    "start": 9,  # 9 AM
    "end": 17,   # 5 PM
    "days": [0, 1, 2, 3, 4]  # Monday to Friday (0=Monday, 6=Sunday)
}
# -----------------------

def is_booking_intent(text: str) -> bool:
    """
    Enhanced booking intent detection with more patterns and better accuracy
    """
    if not text:
        return False
        
    text_lower = text.lower().strip()
    
    # Primary booking keywords with enhanced patterns
    booking_keywords = [
        "book", "appointment", "schedule", "reserve", 
        "meeting", "slot", "time", "available",
        "free time", "book me", "schedule me", "set up",
        "arrange", "fix", "plan", "when can", "can i come"
    ]
    
    # Enhanced time patterns that suggest booking
    time_patterns = [
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",  # 9am, 2:30pm
        r"\btomorrow\b", r"\bnext\s+week\b", r"\btoday\b",
        r"\bthis\s+(?:morning|afternoon|evening|week)\b",
        r"\bmonday|tuesday|wednesday|thursday|friday|saturday|sunday\b",
        r"\bnext\s+(?:monday|tuesday|wednesday|thursday|friday)\b"
    ]
    
    # Action phrases that indicate booking intent
    action_phrases = [
        "need to see", "want to meet", "can we meet", "let's schedule",
        "set something up", "make time", "find a time", "come in"
    ]
    
    # Check for booking keywords
    has_booking_keyword = any(keyword in text_lower for keyword in booking_keywords)
    
    # Check for time patterns
    has_time_pattern = any(re.search(pattern, text_lower) for pattern in time_patterns)
    
    # Check for action phrases
    has_action_phrase = any(phrase in text_lower for phrase in action_phrases)
    
    # Enhanced intent logic
    booking_intent = (
        has_booking_keyword or 
        (has_time_pattern and any(word in text_lower for word in ["can", "could", "would", "want", "need", "i'd like"])) or
        has_action_phrase or
        # Questions about availability
        ("available" in text_lower and any(word in text_lower for word in ["when", "what time", "are you"])) or
        # Direct booking requests
        any(phrase in text_lower for phrase in ["book me", "schedule me", "reserve", "appointment"])
    )
    
    if booking_intent:
        logger.info(f"Booking intent detected in: '{text}'")
        return True
    
    return False

def parse_time_from_text(text: str) -> Optional[datetime]:
    """
    Enhanced time parsing with better error handling and more patterns
    """
    try:
        text_lower = text.lower().strip()
        
        # Pattern 1: Specific times like "9am", "2:30pm", "14:00"
        time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text_lower)
        
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            period = time_match.group(3)
            
            # Convert to 24-hour format
            if period == "pm" and hour < 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
            elif not period and hour < 8:  # Assume PM for hours < 8 without AM/PM
                hour += 12
            
            # Create datetime for today with parsed time
            now = datetime.now()
            slot_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If the time has passed today, schedule for tomorrow
            if slot_time <= now:
                slot_time += timedelta(days=1)
            
            return slot_time
        
        # Pattern 2: Day-specific times like "tomorrow at 2pm", "monday morning"
        day_patterns = {
            'tomorrow': 1,
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4,
            'saturday': 5, 'sunday': 6
        }
        
        for day_name, target_weekday in day_patterns.items():
            if day_name in text_lower:
                if day_name == 'tomorrow':
                    base_time = datetime.now() + timedelta(days=1)
                else:
                    # Find next occurrence of the specified day
                    now = datetime.now()
                    days_ahead = target_weekday - now.weekday()
                    if days_ahead <= 0:  # Target day already happened this week
                        days_ahead += 7
                    base_time = now + timedelta(days=days_ahead)
                
                # Look for specific times
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
                elif "morning" in text_lower:
                    return base_time.replace(hour=9, minute=0, second=0, microsecond=0)
                elif "afternoon" in text_lower:
                    return base_time.replace(hour=14, minute=0, second=0, microsecond=0)
                elif "evening" in text_lower:
                    return base_time.replace(hour=17, minute=0, second=0, microsecond=0)
                else:
                    return base_time.replace(hour=10, minute=0, second=0, microsecond=0)
        
        # Pattern 3: Relative times like "in 2 hours", "next hour"
        if "hour" in text_lower:
            hours_match = re.search(r"(\d+)\s*hour", text_lower)
            if hours_match:
                hours = int(hours_match.group(1))
                return datetime.now() + timedelta(hours=hours)
            elif "next hour" in text_lower:
                return datetime.now() + timedelta(hours=1)
        
        return None
        
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing time from '{text}': {e}")
        return None

def is_valid_business_hours(slot_time: datetime) -> Tuple[bool, str]:
    """
    Check if the requested time is within business hours with enhanced messaging
    """
    # Check day of week (0=Monday, 6=Sunday)
    if slot_time.weekday() not in BUSINESS_HOURS["days"]:
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        requested_day = weekday_names[slot_time.weekday()]
        business_days = [weekday_names[day] for day in BUSINESS_HOURS["days"]]
        return False, f"We're closed on {requested_day}. We're open {', '.join(business_days[:-1])} and {business_days[-1]}."
    
    # Check business hours
    if slot_time.hour < BUSINESS_HOURS["start"]:
        return False, f"We open at {BUSINESS_HOURS['start']}:00 AM. Would you like to schedule something after we open?"
    
    if slot_time.hour >= BUSINESS_HOURS["end"]:
        return False, f"We close at {BUSINESS_HOURS['end']}:00 PM. How about scheduling for the next business day?"
    
    # Check if it's in the past
    if slot_time <= datetime.now():
        return False, "That time has already passed. Please choose a future time."
    
    # Check if it's too far in the future (optional - e.g., 30 days)
    if slot_time > datetime.now() + timedelta(days=30):
        return False, "I can only schedule appointments up to 30 days in advance. Please choose an earlier date."
    
    return True, "Time is valid and within business hours."

async def get_google_sheets_service():
    """
    Create Google Sheets service with proper error handling and retry logic
    """
    try:
        credentials = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, 
            scopes=SCOPES
        )
        service = build("sheets", "v4", credentials=credentials)
        return service
    except FileNotFoundError:
        logger.error(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
        raise Exception(f"Google Sheets configuration file missing: {SERVICE_ACCOUNT_FILE}")
    except Exception as e:
        logger.error(f"Failed to create Google Sheets service: {e}")
        raise Exception("Unable to connect to booking system. Please try again later.")

async def check_slot_availability(service, slot_time: datetime) -> Tuple[bool, str]:
    """
    Check if the requested time slot is available with enhanced error handling
    """
    try:
        # Get existing appointments
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
        booked_times = []
        for row in rows[1:]:  # Skip header row
            if len(row) >= 2:
                if row[0] == date_str and row[1] == time_str:
                    return False, f"Sorry, {slot_time.strftime('%I:%M %p on %A, %B %d')} is already booked. Would you like to try a different time?"
                elif row[0] == date_str:
                    booked_times.append(row[1])
        
        # Suggest alternative times if available
        available_suggestions = []
        for hour in range(BUSINESS_HOURS["start"], BUSINESS_HOURS["end"]):
            for minute in [0, 30]:  # 30-minute slots
                check_time = f"{hour:02d}:{minute:02d}"
                if check_time not in booked_times:
                    check_datetime = datetime.combine(slot_time.date(), datetime.strptime(check_time, "%H:%M").time())
                    if check_datetime > datetime.now():
                        available_suggestions.append(check_datetime.strftime("%I:%M %p"))
                        if len(available_suggestions) >= 3:  # Suggest max 3 alternatives
                            break
            if len(available_suggestions) >= 3:
                break
        
        success_msg = f"Great! {slot_time.strftime('%I:%M %p on %A, %B %d')} is available."
        if available_suggestions:
            success_msg += f" Other available times today include: {', '.join(available_suggestions[:3])}."
        
        return True, success_msg
        
    except HttpError as e:
        logger.error(f"Google Sheets API error: {e}")
        return False, "I'm having trouble checking our booking system right now. Please try again in a moment, or call us directly to book your appointment."
    except Exception as e:
        logger.error(f"Error checking availability: {e}")
        return False, "Technical issue checking availability. Please try again or contact us directly."

async def book_appointment(service, slot_time: datetime, user_id: str, user_name: str = None) -> Tuple[bool, str]:
    """
    Book the appointment in Google Sheets with enhanced confirmation
    """
    try:
        date_str = slot_time.strftime("%Y-%m-%d")
        time_str = slot_time.strftime("%H:%M")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare the new row
        new_row = [
            date_str,           # A: Date
            time_str,           # B: Time
            user_id,            # C: User ID
            user_name or "Voice Assistant User", # D: Name
            "Confirmed",        # E: Status
            timestamp           # F: Created At
        ]
        
        # Append to sheet
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
        success_msg = f"Perfect! Your appointment is confirmed for {formatted_time}. We'll see you then! If you need to reschedule or cancel, please contact us directly."
        
        logger.info(f"Appointment booked successfully: User {user_id} at {formatted_time}")
        return True, success_msg
        
    except HttpError as e:
        logger.error(f"Google Sheets API error during booking: {e}")
        return False, "I couldn't complete your booking due to a technical issue. Please try again, or contact us directly to secure your appointment."
    except Exception as e:
        logger.error(f"Error booking appointment: {e}")
        return False, "Sorry, there was a problem booking your appointment. Please try again or contact us directly."

async def generate_booking_tts_response(text: str) -> Optional[bytes]:
    """
    Generate TTS response for booking with consistent voice
    """
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

async def handle_booking(transcript: str, user_id: str) -> str:
    """
    Enhanced main booking handler with comprehensive error handling and voice consistency
    """
    try:
        logger.info(f"Processing booking request from user {user_id}: '{transcript}'")
        
        # Step 1: Enhanced time parsing with multiple attempts
        slot_time = parse_time_from_text(transcript)
        
        if not slot_time:
            return ("I understand you want to book an appointment, but I need a specific time. "
                   "Please tell me something like 'book me at 2 PM tomorrow' or 'schedule me for 10:30 AM on Monday'. "
                   "I can also help if you say something like 'what times are available tomorrow?'")
        
        logger.info(f"Parsed appointment time: {slot_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Step 2: Enhanced business hours validation
        is_valid, validation_msg = is_valid_business_hours(slot_time)
        if not is_valid:
            business_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            return f"{validation_msg} Our business hours are {BUSINESS_HOURS['start']}:00 AM to {BUSINESS_HOURS['end']}:00 PM, {', '.join(business_days[:-1])} and {business_days[-1]}."
        
        # Step 3: Get Google Sheets service with better error messages
        try:
            service = await get_google_sheets_service()
        except Exception as service_error:
            logger.error(f"Failed to connect to Google Sheets: {service_error}")
            return "I'm having trouble accessing our booking system right now. Please try again in a few minutes, or you can call us directly to schedule your appointment."
        
        # Step 4: Check availability with enhanced feedback
        is_available, availability_msg = await check_slot_availability(service, slot_time)
        if not is_available:
            return f"{availability_msg}"
        
        logger.info(f"Time slot available: {slot_time.strftime('%I:%M %p on %A, %B %d')}")
        
        # Step 5: Book the appointment with confirmation
        booking_success, booking_msg = await book_appointment(service, slot_time, user_id)
        
        if booking_success:
            logger.info(f"Booking successful for user {user_id} at {slot_time}")
            return booking_msg
        else:
            logger.error(f"Booking failed for user {user_id}: {booking_msg}")
            return booking_msg
            
    except Exception as e:
        logger.error(f"Fatal error in booking handler for user {user_id}: {e}", exc_info=True)
        return "I apologize, but I encountered an issue while processing your booking request. Please try again or contact us directly to schedule your appointment."

# Additional helper functions with voice consistency

async def get_available_slots(user_id: str, date: datetime = None) -> str:
    """
    Get available time slots for a specific date with enhanced user experience
    """
    try:
        if date is None:
            date = datetime.now() + timedelta(days=1)  # Default to tomorrow
        
        service = await get_google_sheets_service()
        
        # Get existing appointments for the date
        sheet = service.spreadsheets()
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: sheet.values().get(
                spreadsheetId=SPREADSHEET_ID, 
                range=RANGE_NAME
            ).execute()
        )
        
        rows = result.get("values", [])
        date_str = date.strftime("%Y-%m-%d")
        
        # Get booked times for the date
        booked_times = set()
        for row in rows[1:]:  # Skip header
            if len(row) >= 2 and row[0] == date_str:
                booked_times.add(row[1])
        
        # Generate available slots (30-minute intervals)
        available_slots = []
        for hour in range(BUSINESS_HOURS["start"], BUSINESS_HOURS["end"]):
            for minute in [0, 30]:  # 30-minute slots
                time_str = f"{hour:02d}:{minute:02d}"
                if time_str not in booked_times:
                    slot_datetime = datetime.combine(date.date(), datetime.strptime(time_str, "%H:%M").time())
                    if slot_datetime > datetime.now():
                        available_slots.append(slot_datetime.strftime("%I:%M %p"))
        
        if available_slots:
            day_name = date.strftime('%A, %B %d')
            if len(available_slots) <= 6:
                slots_text = ", ".join(available_slots)
                return f"Here are the available times for {day_name}: {slots_text}. Which time works best for you?"
            else:
                # Show first 6 slots and mention more are available
                slots_text = ", ".join(available_slots[:6])
                return f"Here are some available times for {day_name}: {slots_text}, and more. Which time would you prefer?"
        else:
            return f"I don't see any available slots for {date.strftime('%A, %B %d')}. Would you like to try a different date? I can check tomorrow or any other day that works for you."
            
    except Exception as e:
        logger.error(f"Error getting available slots: {e}")
        return "I'm having trouble checking our availability right now. Please try again in a moment or contact us directly for scheduling."

async def cancel_appointment(user_id: str, appointment_time: datetime = None) -> str:
    """
    Cancel an existing appointment for the user
    """
    try:
        service = await get_google_sheets_service()
        
        # Get existing appointments
        sheet = service.spreadsheets()
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: sheet.values().get(
                spreadsheetId=SPREADSHEET_ID,
                range=RANGE_NAME
            ).execute()
        )
        
        rows = result.get("values", [])
        if not rows or len(rows) <= 1:
            return "I don't find any appointments to cancel. If you have an appointment, please contact us directly."
        
        # Find user's appointments
        user_appointments = []
        for i, row in enumerate(rows[1:], start=2):  # Start from row 2 (skipping header)
            if len(row) >= 3 and row[2] == user_id:
                appointment_datetime = datetime.strptime(f"{row[0]} {row[1]}", "%Y-%m-%d %H:%M")
                if appointment_datetime > datetime.now():  # Only future appointments
                    user_appointments.append({
                        "row": i,
                        "datetime": appointment_datetime,
                        "formatted": appointment_datetime.strftime("%I:%M %p on %A, %B %d")
                    })
        
        if not user_appointments:
            return "I don't find any upcoming appointments for you. If you believe this is an error, please contact us directly."
        
        # If specific time provided, find and cancel that appointment
        if appointment_time:
            target_appointment = None
            for apt in user_appointments:
                if apt["datetime"].date() == appointment_time.date() and apt["datetime"].hour == appointment_time.hour:
                    target_appointment = apt
                    break
            
            if target_appointment:
                # Update the status to "Cancelled"
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: sheet.values().update(
                        spreadsheetId=SPREADSHEET_ID,
                        range=f"E{target_appointment['row']}",
                        valueInputOption="RAW",
                        body={"values": [["Cancelled"]]}
                    ).execute()
                )
                
                return f"Your appointment for {target_appointment['formatted']} has been cancelled. If you need to reschedule, I'm happy to help you find a new time."
            else:
                return f"I couldn't find an appointment at that specific time. You have appointments at: {', '.join([apt['formatted'] for apt in user_appointments])}. Which one would you like to cancel?"
        
        # If no specific time, list appointments and ask which to cancel
        if len(user_appointments) == 1:
            apt = user_appointments[0]
            # Auto-cancel the single appointment
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: sheet.values().update(
                    spreadsheetId=SPREADSHEET_ID,
                    range=f"E{apt['row']}",
                    valueInputOption="RAW",
                    body={"values": [["Cancelled"]]}
                ).execute()
            )
            return f"Your appointment for {apt['formatted']} has been cancelled. If you need to reschedule, I'm happy to help you find a new time."
        else:
            apt_list = ', '.join([apt['formatted'] for apt in user_appointments])
            return f"You have multiple appointments: {apt_list}. Please let me know which specific appointment you'd like to cancel by mentioning the date and time."
    
    except Exception as e:
        logger.error(f"Error cancelling appointment for user {user_id}: {e}")
        return "I'm having trouble accessing our booking system to cancel your appointment. Please contact us directly for cancellation assistance."

async def reschedule_appointment(user_id: str, old_time: datetime, new_time: datetime) -> str:
    """
    Reschedule an existing appointment
    """
    try:
        # First, validate the new time
        is_valid, validation_msg = is_valid_business_hours(new_time)
        if not is_valid:
            return f"The new time isn't available. {validation_msg}"
        
        service = await get_google_sheets_service()
        
        # Check if new slot is available
        is_available, availability_msg = await check_slot_availability(service, new_time)
        if not is_available:
            return f"The new time slot isn't available. {availability_msg}"
        
        # Find and update the existing appointment
        sheet = service.spreadsheets()
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: sheet.values().get(
                spreadsheetId=SPREADSHEET_ID,
                range=RANGE_NAME
            ).execute()
        )
        
        rows = result.get("values", [])
        
        old_date_str = old_time.strftime("%Y-%m-%d")
        old_time_str = old_time.strftime("%H:%M")
        
        for i, row in enumerate(rows[1:], start=2):  # Start from row 2
            if (len(row) >= 3 and row[2] == user_id and 
                row[0] == old_date_str and row[1] == old_time_str):
                
                # Update the appointment with new date and time
                new_date_str = new_time.strftime("%Y-%m-%d")
                new_time_str = new_time.strftime("%H:%M")
                
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: sheet.values().update(
                        spreadsheetId=SPREADSHEET_ID,
                        range=f"A{i}:B{i}",
                        valueInputOption="RAW",
                        body={"values": [[new_date_str, new_time_str]]}
                    ).execute()
                )
                
                old_formatted = old_time.strftime("%I:%M %p on %A, %B %d")
                new_formatted = new_time.strftime("%I:%M %p on %A, %B %d")
                
                return f"Perfect! I've rescheduled your appointment from {old_formatted} to {new_formatted}. We'll see you at the new time!"
        
        return "I couldn't find your original appointment to reschedule. Please contact us directly for assistance with rescheduling."
        
    except Exception as e:
        logger.error(f"Error rescheduling appointment for user {user_id}: {e}")
        return "I'm having trouble rescheduling your appointment. Please contact us directly for assistance."

def detect_booking_action(text: str) -> str:
    """
    Detect specific booking actions from user input
    Returns: 'book', 'cancel', 'reschedule', 'check_availability', or 'unknown'
    """
    text_lower = text.lower().strip()
    
    # Cancellation keywords
    if any(word in text_lower for word in ['cancel', 'cancelled', 'delete', 'remove', 'not coming', "can't make it"]):
        return 'cancel'
    
    # Rescheduling keywords
    if any(phrase in text_lower for phrase in ['reschedule', 'move', 'change', 'different time', 'new time']):
        return 'reschedule'
    
    # Availability check
    if any(phrase in text_lower for phrase in ['available', 'free', 'what times', 'when can', 'open slots']):
        return 'check_availability'
    
    # Booking keywords (default)
    if any(word in text_lower for word in ['book', 'schedule', 'appointment', 'reserve', 'meet']):
        return 'book'
    
    return 'unknown'

async def handle_booking_with_action_detection(transcript: str, user_id: str) -> str:
    """
    Enhanced booking handler that detects different booking-related actions
    """
    try:
        action = detect_booking_action(transcript)
        logger.info(f"Detected booking action: {action} for user {user_id}")
        
        if action == 'check_availability':
            # Parse date from transcript if provided
            requested_date = parse_time_from_text(transcript)
            if requested_date:
                return await get_available_slots(user_id, requested_date)
            else:
                return await get_available_slots(user_id)
        
        elif action == 'cancel':
            # Parse time if user specified which appointment to cancel
            appointment_time = parse_time_from_text(transcript)
            return await cancel_appointment(user_id, appointment_time)
        
        elif action == 'reschedule':
            # This is more complex - would need to parse both old and new times
            # For now, provide helpful message
            return ("I can help you reschedule your appointment. Please tell me your current appointment time and "
                   "when you'd like to move it to. For example: 'move my 2 PM Monday appointment to 3 PM Tuesday'.")
        
        elif action == 'book' or action == 'unknown':
            # Default to standard booking flow
            return await handle_booking(transcript, user_id)
        
    except Exception as e:
        logger.error(f"Error in booking action detection for user {user_id}: {e}")
        return await handle_booking(transcript, user_id)  # Fallback to standard booking

# Test function with voice consistency
async def test_booking_system():
    """Test the enhanced booking system with voice consistency"""
    test_user = "test_user_123"
    test_queries = [
        "book me at 2 pm tomorrow",
        "schedule appointment for 9:30 am on Monday", 
        "what times are available tomorrow?",
        "can I get a slot at 5 pm",
        "cancel my appointment",
        "reschedule my meeting",
        "book me for 25:00",  # Invalid time
        "are you free this afternoon?"
    ]
    
    print("Testing Enhanced Booking System with Voice Consistency:")
    print(f"Using TTS Voice: {TTS_VOICE}")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Test booking intent detection
        is_booking = is_booking_intent(query)
        print(f"Booking Intent Detected: {is_booking}")
        
        if is_booking:
            # Test action detection
            action = detect_booking_action(query)
            print(f"Detected Action: {action}")
            
            # Test full booking handler
            result = await handle_booking_with_action_detection(query, test_user)
            print(f"Response: {result}")
            
            # Test TTS generation (voice consistency)
            voice_data = await generate_booking_tts_response(result)
            if voice_data:
                print(f"TTS Generated: {len(voice_data)} bytes with voice {TTS_VOICE}")
            else:
                print("TTS Generation Failed")
        else:
            print("Not a booking query - would be handled by main AI agent")
        
        print("-" * 30)

# Integration helpers for main app
async def is_booking_query(transcript: str) -> bool:
    """Simple wrapper for main app integration"""
    return is_booking_intent(transcript)

async def process_booking_query(transcript: str, user_id: str) -> str:
    """Main integration point for app.py"""
    return await handle_booking_with_action_detection(transcript, user_id)

# Voice consistency helper
async def get_booking_response_with_voice(transcript: str, user_id: str) -> dict:
    """
    Get booking response with TTS audio using consistent voice
    Returns dict with text, voice_data, and voice info
    """
    try:
        text_response = await process_booking_query(transcript, user_id)
        voice_data = await generate_booking_tts_response(text_response)
        
        return {
            "text": text_response,
            "voice_data": voice_data,
            "voice_used": TTS_VOICE,
            "consistent_voice": True,
            "booking_processed": True
        }
    except Exception as e:
        logger.error(f"Error generating booking response with voice: {e}")
        return {
            "text": "I'm having trouble with the booking system. Please try again or contact us directly.",
            "voice_data": None,
            "voice_used": TTS_VOICE,
            "consistent_voice": True,
            "booking_processed": False
        }

if __name__ == "__main__":
    # Run enhanced tests
    print("Enhanced Booking Module - Voice Consistent & Feature Rich")
    print("Features:")
    print("- Enhanced booking intent detection")
    print("- Multiple booking actions (book, cancel, reschedule, check availability)")
    print("- Consistent voice with main system (en-IN-NeerjaNeural)")
    print("- Better error handling and user feedback")
    print("- Integration ready for main app")
    
    asyncio.run(test_booking_system())