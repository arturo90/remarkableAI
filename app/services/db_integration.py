"""Database integration helper for Gmail service."""
from sqlalchemy.orm import Session
from app.db.base import SessionLocal
from app.db.services import (
    NoteService, TaskService, TopicService, ImportantDateService
)
from app.db.models import User
from datetime import datetime
import hashlib
from pathlib import Path


def save_note_to_database(
    message_id: str,
    attachment_id: str,
    subject: str,
    filename: str,
    file_path: str,
    from_email: str = None,
    received_at: datetime = None,
    file_size: int = None,
    file_data: bytes = None,
) -> int:
    """Save a note to the database and return note ID."""
    db = SessionLocal()
    try:
        # Get default user
        default_user = db.query(User).filter(User.email == "default@remarkableai.com").first()
        if not default_user:
            raise ValueError("Default user not found")
        
        # Calculate file hash if file_data is provided
        file_hash = None
        if file_data:
            file_hash = hashlib.md5(file_data).hexdigest()
        elif Path(file_path).exists():
            file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
        
        # Check if note already exists
        existing_note = NoteService.get_note_by_message_id(db, message_id)
        if existing_note:
            return existing_note.id
        
        # Check for duplicate by hash
        if file_hash:
            existing_note_by_hash = NoteService.get_note_by_hash(db, file_hash)
            if existing_note_by_hash:
                return existing_note_by_hash.id
        
        # Get file size if not provided
        if not file_size and Path(file_path).exists():
            file_size = Path(file_path).stat().st_size
        
        # Create note
        note = NoteService.create_note(
            db=db,
            message_id=message_id,
            attachment_id=attachment_id,
            subject=subject,
            filename=filename,
            file_path=file_path,
            from_email=from_email,
            received_at=received_at,
            file_size=file_size,
            file_hash=file_hash,
            user_id=default_user.id,
        )
        
        return note.id
    finally:
        db.close()


def save_ai_result_to_database(
    message_id: str,
    attachment_id: str,
    ai_result: dict,
    processing_method: str = "unknown",
) -> int:
    """Save AI processing results to database and return note ID."""
    db = SessionLocal()
    try:
        # Find note
        note = NoteService.get_note_by_message_id(db, message_id)
        if not note:
            raise ValueError(f"Note not found for message_id: {message_id}")
        
        # Update note with AI results
        NoteService.update_note_processing_result(
            db=db,
            note_id=note.id,
            summary=ai_result.get('summary'),
            transcription=ai_result.get('transcription'),
            raw_text=ai_result.get('raw_text', ''),
            confidence=ai_result.get('confidence', 0.0),
            ai_result_json=ai_result,
            processing_method=processing_method,
            processing_status="completed",
        )
        
        # Create tasks from AI result
        if 'tasks' in ai_result and ai_result['tasks']:
            TaskService.create_tasks_from_ai_result(
                db=db,
                note_id=note.id,
                tasks=ai_result['tasks'],
                user_id=note.user_id,
            )
        
        # Create topics from AI result
        if 'topics' in ai_result and ai_result['topics']:
            TopicService.create_topics_from_ai_result(
                db=db,
                note_id=note.id,
                topics=ai_result['topics'],
                user_id=note.user_id,
            )
        
        # Create dates from AI result
        if 'dates' in ai_result and ai_result['dates']:
            ImportantDateService.create_dates_from_ai_result(
                db=db,
                note_id=note.id,
                dates=ai_result['dates'],
                user_id=note.user_id,
            )
        
        return note.id
    finally:
        db.close()

