from fastapi import Request, Response
import secrets
import time
from typing import Dict, Any

# In-memory session store (for development - use Redis in production)
_session_store: Dict[str, Dict[str, Any]] = {}

SESSION_COOKIE = 'remarkable_session'

# Generate a session secret for signing session IDs
SESSION_SECRET = secrets.token_hex(32)
print(f"Generated session secret: {SESSION_SECRET[:10]}...")  # Debug

SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 1 week

def generate_session_id() -> str:
    """Generate a unique session ID."""
    return secrets.token_hex(16)

def set_session(response: Response, data: dict):
    """Set session data using a simple session ID cookie."""
    session_id = generate_session_id()
    _session_store[session_id] = {
        'data': data,
        'created_at': time.time()
    }
    print(f"Setting session with ID: {session_id[:8]}...")  # Debug
    print(f"Session data keys: {list(data.keys())}")  # Debug
    
    # Set a simple session ID cookie with minimal configuration
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=SESSION_MAX_AGE,
        path='/',
    )
    print(f"Session ID cookie set: {session_id[:8]}...")  # Debug

def get_session(request: Request) -> dict:
    """Get session data from session ID."""
    session_id = request.cookies.get(SESSION_COOKIE)
    print(f"Getting session, ID present: {bool(session_id)}")  # Debug
    
    if not session_id:
        return {}
    
    session_data = _session_store.get(session_id)
    if not session_data:
        print(f"Session ID not found in store: {session_id[:8]}...")  # Debug
        return {}
    
    # Check if session has expired
    if time.time() - session_data['created_at'] > SESSION_MAX_AGE:
        print(f"Session expired: {session_id[:8]}...")  # Debug
        del _session_store[session_id]
        return {}
    
    print(f"Session found with keys: {list(session_data['data'].keys())}")  # Debug
    return session_data['data']

def clear_session(response: Response):
    """Clear session data."""
    session_id = response.cookies.get(SESSION_COOKIE)
    if session_id and session_id in _session_store:
        del _session_store[session_id]
    response.delete_cookie(SESSION_COOKIE, path='/')
    print("Session cleared")  # Debug 