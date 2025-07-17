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
from app.core.session import set_session, get_session, clear_session
import os

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

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/pdfs")
async def pdfs_page(request: Request):
    """PDF management page."""
    return templates.TemplateResponse("pdfs.html", {"request": request})

@app.get("/results")
async def results_page(request: Request):
    """Results viewer page."""
    return templates.TemplateResponse("results.html", {"request": request})

@app.get("/settings")
async def settings_page(request: Request):
    """Settings page."""
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/upload")
async def upload_page(request: Request):
    """PDF upload page."""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/login")
def login():
    return RedirectResponse(get_auth_url())

@app.get("/auth/callback")
def auth_callback(request: Request, response: Response, code: str = Query(...)):
    try:
        tokens = exchange_code_for_tokens(code)
        set_session(response, {"google_tokens": tokens})
        return RedirectResponse("/gmail_profile", status_code=status.HTTP_302_FOUND)
    except Exception:
        return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

@app.get("/gmail_profile")
def gmail_profile(request: Request, response: Response):
    session = get_session(request)
    tokens = session.get("google_tokens")
    profile, new_tokens = gmail_api_get("users/me/profile", tokens)
    if new_tokens != tokens:
        set_session(response, {"google_tokens": new_tokens})
    if not profile:
        return RedirectResponse("/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("profile.html", {"request": request, "profile": profile})

@app.get("/logout")
def logout(request: Request, response: Response):
    session = get_session(request)
    tokens = session.get("google_tokens")
    revoke_tokens(tokens)
    clear_session(response)
    return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

@app.get("/calendar/connect")
def calendar_connect():
    return RedirectResponse(get_calendar_auth_url())

@app.get("/calendar/list")
def calendar_list(request: Request, response: Response):
    session = get_session(request)
    tokens = session.get("google_tokens")
    calendars, new_tokens = calendar_list_calendars(tokens)
    if new_tokens != tokens:
        set_session(response, {"google_tokens": new_tokens})
    if not calendars:
        return RedirectResponse("/calendar/connect", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("calendar_list.html", {"request": request, "calendars": calendars.get('items', [])})

@app.get("/pdf_tasks")
def pdf_tasks(request: Request):
    # Assume get_pdf_tasks() returns a list of dicts with 'title', 'description', 'due_date'
    from app.services.pdf_service import get_pdf_tasks
    tasks = get_pdf_tasks()
    return templates.TemplateResponse("pdfs.html", {"request": request, "tasks": tasks})

@app.post("/calendar/add_task")
def add_task_to_calendar(request: Request, response: Response, calendar_id: str = Form(...), title: str = Form(...), description: str = Form(...), due_date: str = Form(...)):
    session = get_session(request)
    tokens = session.get("google_tokens")
    event = {
        "summary": title,
        "description": description,
        "start": {"dateTime": due_date, "timeZone": "UTC"},
        "end": {"dateTime": due_date, "timeZone": "UTC"},
    }
    created, new_tokens = calendar_create_event(tokens, calendar_id, event)
    if new_tokens != tokens:
        set_session(response, {"google_tokens": new_tokens})
    return RedirectResponse("/calendar/list", status_code=status.HTTP_302_FOUND) 