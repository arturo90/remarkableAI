from fastapi import FastAPI, Request, Response, status, Query, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import gmail
from app.services.gmail_service import (
    get_auth_url, exchange_code_for_tokens, gmail_api_get, revoke_tokens,
    get_calendar_auth_url, calendar_list_calendars, calendar_create_event
)
import os
import time

app = FastAPI(
    title="RemarkableAI",
    description="An intelligent note processing system for Remarkable tablet notes",
    version="1.0.0"
)

# Configure templates
templates = Jinja2Templates(directory="app/templates")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(gmail.router)

# Simple in-memory token store (for development)
_tokens = None
_user_email = None

def get_simple_auth_context():
    """Get simple authentication context."""
    return {
        "user_connected": bool(_tokens),
        "user_email": _user_email,
        "calendar_connected": False,  # Simplified for now
    }

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        **get_simple_auth_context()
    })

@app.get("/auth")
def auth_page(request: Request):
    """Simple authentication page."""
    if _tokens:
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    
    return templates.TemplateResponse("auth.html", {
        "request": request,
        **get_simple_auth_context()
    })

@app.get("/auth/callback")
def auth_callback(request: Request, response: Response, code: str = Query(...)):
    global _tokens, _user_email
    try:
        print("Exchanging code for tokens...")  # Debug
        tokens = exchange_code_for_tokens(code)
        print(f"Tokens received: {list(tokens.keys())}")  # Debug
        
        # Store tokens globally
        _tokens = tokens
        
        # Get user email
        try:
            profile, _ = gmail_api_get("users/me/profile", tokens)
            if profile and "emailAddress" in profile:
                _user_email = profile["emailAddress"]
                print(f"User email: {_user_email}")  # Debug
        except Exception as e:
            print(f"Error getting profile: {e}")
            _user_email = "user@gmail.com"  # Fallback
        
        print("Authentication successful, redirecting to dashboard")  # Debug
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    except Exception as e:
        print(f"Authentication error: {e}")  # Debug
        return RedirectResponse("/auth", status_code=status.HTTP_302_FOUND)

@app.get("/login")
def login():
    return RedirectResponse(get_auth_url())

@app.get("/logout")
def logout(request: Request, response: Response):
    global _tokens, _user_email
    if _tokens:
        revoke_tokens(_tokens)
    _tokens = None
    _user_email = None
    print("Logged out")  # Debug
    return RedirectResponse("/auth", status_code=status.HTTP_302_FOUND)

@app.get("/pdfs")
async def pdfs_page(request: Request):
    """PDF management page."""
    return templates.TemplateResponse("pdfs.html", {
        "request": request,
        **get_simple_auth_context()
    })

@app.get("/results")
async def results_page(request: Request):
    """Results viewer page."""
    return templates.TemplateResponse("results.html", {
        "request": request,
        **get_simple_auth_context()
    })

@app.get("/settings")
async def settings_page(request: Request):
    """Settings page."""
    return templates.TemplateResponse("settings.html", {
        "request": request,
        **get_simple_auth_context()
    })

@app.get("/upload")
async def upload_page(request: Request):
    """PDF upload page."""
    return templates.TemplateResponse("upload.html", {
        "request": request,
        **get_simple_auth_context()
    })

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/calendar/connect")
def calendar_connect():
    return RedirectResponse(get_calendar_auth_url())

@app.get("/calendar/list")
def calendar_list(request: Request, response: Response):
    global _tokens
    if not _tokens:
        return RedirectResponse("/auth", status_code=status.HTTP_302_FOUND)
    
    calendars, new_tokens = calendar_list_calendars(_tokens)
    if new_tokens != _tokens:
        _tokens = new_tokens
    
    if not calendars:
        return RedirectResponse("/calendar/connect", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("calendar_list.html", {
        "request": request, 
        "calendars": calendars.get('items', [])
    })

@app.get("/pdf_tasks")
def pdf_tasks(request: Request):
    from app.services.pdf_service import get_pdf_tasks
    tasks = get_pdf_tasks()
    return templates.TemplateResponse("pdfs.html", {
        "request": request, 
        "tasks": tasks
    })

@app.post("/calendar/add_task")
def add_task_to_calendar(request: Request, response: Response, calendar_id: str = Form(...), title: str = Form(...), description: str = Form(...), due_date: str = Form(...)):
    global _tokens
    if not _tokens:
        return RedirectResponse("/auth", status_code=status.HTTP_302_FOUND)
    
    event = {
        "summary": title,
        "description": description,
        "start": {"dateTime": due_date, "timeZone": "UTC"},
        "end": {"dateTime": due_date, "timeZone": "UTC"},
    }
    created, new_tokens = calendar_create_event(_tokens, calendar_id, event)
    if new_tokens != _tokens:
        _tokens = new_tokens
    return RedirectResponse("/calendar/list", status_code=status.HTTP_302_FOUND)

@app.get("/test-cookie")
def test_cookie(request: Request, response: Response):
    """Test endpoint to check if cookies are working."""
    test_value = "test_cookie_value"
    response.set_cookie("test_cookie", test_value, path='/')
    
    received_cookies = list(request.cookies.keys())
    test_cookie_value = request.cookies.get("test_cookie")
    
    return {
        "message": "Cookie test",
        "cookies_sent": received_cookies,
        "test_cookie_present": bool(test_cookie_value),
        "test_cookie_value": test_cookie_value
    } 