"""API endpoints for notes management."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.base import get_db
from app.db.services import NoteService
from app.db.models import Note

router = APIRouter(prefix="/api/notes", tags=["notes"])

@router.get("/", response_model=List[dict])
async def get_notes(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get all notes with pagination."""
    try:
        notes = NoteService.get_notes(
            db=db,
            user_id=1,  # Default user for now
            skip=skip,
            limit=limit,
            status=status,
        )
        return [
            {
                "id": note.id,
                "message_id": note.message_id,
                "attachment_id": note.attachment_id,
                "subject": note.subject,
                "filename": note.filename,
                "file_path": note.file_path,
                "from_email": note.from_email,
                "received_at": note.received_at.isoformat() if note.received_at else None,
                "processed_at": note.processed_at.isoformat() if note.processed_at else None,
                "processing_status": note.processing_status,
                "processing_method": note.processing_method,
                "summary": note.summary,
                "confidence": note.confidence,
                "created_at": note.created_at.isoformat() if note.created_at else None,
                "task_count": len(note.tasks),
                "topic_count": len(note.topics),
                "date_count": len(note.dates),
            }
            for note in notes
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{note_id}", response_model=dict)
async def get_note(
    note_id: int,
    db: Session = Depends(get_db),
):
    """Get a specific note by ID."""
    try:
        note = NoteService.get_note_by_id(db=db, note_id=note_id)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        
        return {
            "id": note.id,
            "message_id": note.message_id,
            "attachment_id": note.attachment_id,
            "subject": note.subject,
            "filename": note.filename,
            "file_path": note.file_path,
            "from_email": note.from_email,
            "received_at": note.received_at.isoformat() if note.received_at else None,
            "processed_at": note.processed_at.isoformat() if note.processed_at else None,
            "processing_status": note.processing_status,
            "processing_method": note.processing_method,
            "processing_error": note.processing_error,
            "summary": note.summary,
            "transcription": note.transcription,
            "raw_text": note.raw_text,
            "confidence": note.confidence,
            "ai_result_json": note.ai_result_json,
            "created_at": note.created_at.isoformat() if note.created_at else None,
            "updated_at": note.updated_at.isoformat() if note.updated_at else None,
            "tasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "status": task.status,
                    "priority": task.priority,
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                }
                for task in note.tasks
            ],
            "topics": [
                {
                    "id": topic.id,
                    "name": topic.name,
                    "description": topic.description,
                }
                for topic in note.topics
            ],
            "dates": [
                {
                    "id": date.id,
                    "date": date.date.isoformat() if date.date else None,
                    "date_text": date.date_text,
                    "description": date.description,
                    "date_type": date.date_type,
                }
                for date in note.dates
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/{query}", response_model=List[dict])
async def search_notes(
    query: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Search notes by text."""
    try:
        notes = NoteService.search_notes(
            db=db,
            query=query,
            user_id=1,  # Default user for now
            skip=skip,
            limit=limit,
        )
        return [
            {
                "id": note.id,
                "message_id": note.message_id,
                "subject": note.subject,
                "filename": note.filename,
                "summary": note.summary,
                "confidence": note.confidence,
                "created_at": note.created_at.isoformat() if note.created_at else None,
            }
            for note in notes
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{note_id}")
async def delete_note(
    note_id: int,
    db: Session = Depends(get_db),
):
    """Delete a note."""
    try:
        success = NoteService.delete_note(db=db, note_id=note_id)
        if not success:
            raise HTTPException(status_code=404, detail="Note not found")
        return {"message": "Note deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

