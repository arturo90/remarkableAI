from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.api import gmail

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
async def root(request: Request):
    """Root endpoint to serve the dashboard."""
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