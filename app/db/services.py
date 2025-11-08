"""Database service layer for RemarkableAI."""
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc, func, extract
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import json
from app.db.models import (
    User, Note, Task, Topic, ImportantDate,
    TaskStatus, TaskPriority
)
from app.db.base import Base


class NoteService:
    """Service for managing notes."""
    
    @staticmethod
    def create_note(
        db: Session,
        message_id: str,
        attachment_id: str,
        subject: str,
        filename: str,
        file_path: str,
        from_email: Optional[str] = None,
        received_at: Optional[datetime] = None,
        file_size: Optional[int] = None,
        file_hash: Optional[str] = None,
        user_id: int = 1,
    ) -> Note:
        """Create a new note."""
        note = Note(
            message_id=message_id,
            attachment_id=attachment_id,
            subject=subject,
            filename=filename,
            file_path=file_path,
            from_email=from_email,
            received_at=received_at,
            file_size=file_size,
            file_hash=file_hash,
            user_id=user_id,
            processing_status="pending",
        )
        db.add(note)
        db.commit()
        db.refresh(note)
        return note
    
    @staticmethod
    def get_note_by_id(db: Session, note_id: int) -> Optional[Note]:
        """Get note by ID."""
        return db.query(Note).filter(Note.id == note_id).first()
    
    @staticmethod
    def get_note_by_message_id(db: Session, message_id: str) -> Optional[Note]:
        """Get note by Gmail message ID."""
        return db.query(Note).filter(Note.message_id == message_id).first()
    
    @staticmethod
    def get_note_by_hash(db: Session, file_hash: str) -> Optional[Note]:
        """Get note by file hash (for deduplication)."""
        return db.query(Note).filter(Note.file_hash == file_hash).first()
    
    @staticmethod
    def get_notes(
        db: Session,
        user_id: int = 1,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
    ) -> List[Note]:
        """Get all notes with pagination."""
        query = db.query(Note).filter(Note.user_id == user_id)
        
        if status:
            query = query.filter(Note.processing_status == status)
        
        return query.order_by(desc(Note.created_at)).offset(skip).limit(limit).all()
    
    @staticmethod
    def update_note_processing_result(
        db: Session,
        note_id: int,
        summary: Optional[str] = None,
        transcription: Optional[str] = None,
        raw_text: Optional[str] = None,
        confidence: Optional[float] = None,
        ai_result_json: Optional[Dict[str, Any]] = None,
        processing_method: Optional[str] = None,
        processing_status: str = "completed",
    ) -> Optional[Note]:
        """Update note with AI processing results."""
        note = db.query(Note).filter(Note.id == note_id).first()
        if not note:
            return None
        
        if summary is not None:
            note.summary = summary
        if transcription is not None:
            note.transcription = transcription
        if raw_text is not None:
            note.raw_text = raw_text
        if confidence is not None:
            note.confidence = confidence
        if ai_result_json is not None:
            note.ai_result_json = ai_result_json
        if processing_method is not None:
            note.processing_method = processing_method
        
        note.processing_status = processing_status
        note.processed_at = datetime.utcnow()
        
        db.commit()
        db.refresh(note)
        return note
    
    @staticmethod
    def delete_note(db: Session, note_id: int) -> bool:
        """Delete a note and all related data."""
        note = db.query(Note).filter(Note.id == note_id).first()
        if not note:
            return False
        
        db.delete(note)
        db.commit()
        return True
    
    @staticmethod
    def search_notes(
        db: Session,
        query: str,
        user_id: int = 1,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Note]:
        """Search notes by text (simple LIKE search for now)."""
        search_term = f"%{query}%"
        return db.query(Note).filter(
            and_(
                Note.user_id == user_id,
                or_(
                    Note.subject.ilike(search_term),
                    Note.summary.ilike(search_term),
                    Note.transcription.ilike(search_term),
                    Note.raw_text.ilike(search_term),
                )
            )
        ).order_by(desc(Note.created_at)).offset(skip).limit(limit).all()


class TaskService:
    """Service for managing tasks."""
    
    @staticmethod
    def _generate_task_hash(title: str, note_id: int) -> str:
        """Generate hash for task deduplication."""
        hash_string = f"{title.lower().strip()}_{note_id}"
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    @staticmethod
    def create_task(
        db: Session,
        note_id: int,
        title: str,
        description: Optional[str] = None,
        status: TaskStatus = TaskStatus.PENDING,
        priority: TaskPriority = TaskPriority.MEDIUM,
        due_date: Optional[datetime] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_text: Optional[str] = None,
        confidence: float = 0.0,
        user_id: int = 1,
    ) -> Task:
        """Create a new task."""
        task_hash = TaskService._generate_task_hash(title, note_id)
        
        # Check for duplicates
        existing_task = db.query(Task).filter(
            and_(
                Task.task_hash == task_hash,
                Task.note_id == note_id,
            )
        ).first()
        
        if existing_task:
            # Mark as duplicate
            task = Task(
                note_id=note_id,
                user_id=user_id,
                title=title,
                description=description,
                status=status.value if isinstance(status, TaskStatus) else status,
                priority=priority.value if isinstance(priority, TaskPriority) else priority,
                due_date=due_date,
                category=category,
                tags=tags,
                source_text=source_text,
                confidence=confidence,
                task_hash=task_hash,
                is_duplicate=True,
                original_task_id=existing_task.id,
            )
        else:
            task = Task(
                note_id=note_id,
                user_id=user_id,
                title=title,
                description=description,
                status=status.value if isinstance(status, TaskStatus) else status,
                priority=priority.value if isinstance(priority, TaskPriority) else priority,
                due_date=due_date,
                category=category,
                tags=tags,
                source_text=source_text,
                confidence=confidence,
                task_hash=task_hash,
                is_duplicate=False,
            )
        
        db.add(task)
        db.commit()
        db.refresh(task)
        return task
    
    @staticmethod
    def create_tasks_from_ai_result(
        db: Session,
        note_id: int,
        tasks: List[str],
        user_id: int = 1,
    ) -> List[Task]:
        """Create multiple tasks from AI result."""
        created_tasks = []
        for task_text in tasks:
            if task_text and task_text.strip():
                # Clean up task text
                task_title = task_text.strip()
                # Remove common prefixes
                for prefix in ['•', '-', '*', '→', '>', 'todo:', 'task:', 'action:']:
                    if task_title.lower().startswith(prefix.lower()):
                        task_title = task_title[len(prefix):].strip()
                
                if task_title:
                    task = TaskService.create_task(
                        db=db,
                        note_id=note_id,
                        title=task_title,
                        source_text=task_text,
                        user_id=user_id,
                    )
                    created_tasks.append(task)
        
        return created_tasks
    
    @staticmethod
    def get_task_by_id(db: Session, task_id: int) -> Optional[Task]:
        """Get task by ID."""
        return db.query(Task).filter(Task.id == task_id).first()
    
    @staticmethod
    def get_all_tasks(
        db: Session,
        user_id: int = 1,
        skip: int = 0,
        limit: int = 100,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
        include_duplicates: bool = False,
    ) -> List[Task]:
        """Get all tasks with filters."""
        query = db.query(Task).filter(Task.user_id == user_id)
        
        if not include_duplicates:
            query = query.filter(Task.is_duplicate == False)
        
        if status:
            query = query.filter(Task.status == status)
        
        if priority:
            query = query.filter(Task.priority == priority)
        
        return query.order_by(desc(Task.created_at)).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_tasks_by_note(db: Session, note_id: int) -> List[Task]:
        """Get all tasks for a specific note."""
        return db.query(Task).filter(Task.note_id == note_id).all()
    
    @staticmethod
    def update_task_status(
        db: Session,
        task_id: int,
        status: TaskStatus,
    ) -> Optional[Task]:
        """Update task status."""
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return None
        
        task.status = status.value if isinstance(status, TaskStatus) else status
        if status == TaskStatus.COMPLETED:
            task.completed_at = datetime.utcnow()
        else:
            task.completed_at = None
        
        db.commit()
        db.refresh(task)
        return task
    
    @staticmethod
    def update_task_priority(
        db: Session,
        task_id: int,
        priority: TaskPriority,
    ) -> Optional[Task]:
        """Update task priority."""
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return None
        
        task.priority = priority.value if isinstance(priority, TaskPriority) else priority
        db.commit()
        db.refresh(task)
        return task
    
    @staticmethod
    def update_task_due_date(
        db: Session,
        task_id: int,
        due_date: Optional[datetime],
    ) -> Optional[Task]:
        """Update task due date."""
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return None
        
        task.due_date = due_date
        db.commit()
        db.refresh(task)
        return task
    
    @staticmethod
    def delete_task(db: Session, task_id: int) -> bool:
        """Delete a task."""
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return False
        
        db.delete(task)
        db.commit()
        return True
    
    @staticmethod
    def search_tasks(
        db: Session,
        query: str,
        user_id: int = 1,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Task]:
        """Search tasks by text."""
        search_term = f"%{query}%"
        return db.query(Task).filter(
            and_(
                Task.user_id == user_id,
                Task.is_duplicate == False,
                or_(
                    Task.title.ilike(search_term),
                    Task.description.ilike(search_term),
                    Task.source_text.ilike(search_term),
                )
            )
        ).order_by(desc(Task.created_at)).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_task_statistics(db: Session, user_id: int = 1) -> Dict[str, Any]:
        """Get task statistics."""
        total = db.query(Task).filter(
            and_(Task.user_id == user_id, Task.is_duplicate == False)
        ).count()
        
        pending = db.query(Task).filter(
            and_(
                Task.user_id == user_id,
                Task.is_duplicate == False,
                Task.status == TaskStatus.PENDING
            )
        ).count()
        
        completed = db.query(Task).filter(
            and_(
                Task.user_id == user_id,
                Task.is_duplicate == False,
                Task.status == TaskStatus.COMPLETED
            )
        ).count()
        
        overdue = db.query(Task).filter(
            and_(
                Task.user_id == user_id,
                Task.is_duplicate == False,
                Task.status != TaskStatus.COMPLETED,
                Task.due_date < datetime.utcnow()
            )
        ).count()
        
        return {
            "total": total,
            "pending": pending,
            "completed": completed,
            "overdue": overdue,
            "completion_rate": (completed / total * 100) if total > 0 else 0,
        }


class TopicService:
    """Service for managing topics."""
    
    @staticmethod
    def create_topic(
        db: Session,
        note_id: int,
        name: str,
        description: Optional[str] = None,
        relevance_score: float = 0.0,
        user_id: int = 1,
    ) -> Topic:
        """Create a new topic."""
        topic = Topic(
            note_id=note_id,
            user_id=user_id,
            name=name,
            description=description,
            relevance_score=relevance_score,
        )
        db.add(topic)
        db.commit()
        db.refresh(topic)
        return topic
    
    @staticmethod
    def create_topics_from_ai_result(
        db: Session,
        note_id: int,
        topics: List[str],
        user_id: int = 1,
    ) -> List[Topic]:
        """Create multiple topics from AI result."""
        created_topics = []
        for topic_text in topics:
            if topic_text and topic_text.strip():
                topic_name = topic_text.strip()
                # Remove common prefixes
                for prefix in ['•', '-', '*', '→', '>', 'topic:', 'subject:', 'theme:']:
                    if topic_name.lower().startswith(prefix.lower()):
                        topic_name = topic_name[len(prefix):].strip()
                
                if topic_name:
                    topic = TopicService.create_topic(
                        db=db,
                        note_id=note_id,
                        name=topic_name,
                        user_id=user_id,
                    )
                    created_topics.append(topic)
        
        return created_topics
    
    @staticmethod
    def get_topics_by_note(db: Session, note_id: int) -> List[Topic]:
        """Get all topics for a specific note."""
        return db.query(Topic).filter(Topic.note_id == note_id).all()
    
    @staticmethod
    def get_all_topics(
        db: Session,
        user_id: int = 1,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Topic]:
        """Get all topics."""
        return db.query(Topic).filter(Topic.user_id == user_id).order_by(
            desc(Topic.relevance_score)
        ).offset(skip).limit(limit).all()


class ImportantDateService:
    """Service for managing important dates."""
    
    @staticmethod
    def create_date(
        db: Session,
        note_id: int,
        date: datetime,
        date_text: Optional[str] = None,
        description: Optional[str] = None,
        date_type: Optional[str] = None,
        confidence: float = 0.0,
        user_id: int = 1,
    ) -> ImportantDate:
        """Create a new important date."""
        important_date = ImportantDate(
            note_id=note_id,
            user_id=user_id,
            date=date,
            date_text=date_text,
            description=description,
            date_type=date_type,
            confidence=confidence,
        )
        db.add(important_date)
        db.commit()
        db.refresh(important_date)
        return important_date
    
    @staticmethod
    def create_dates_from_ai_result(
        db: Session,
        note_id: int,
        dates: List[str],
        user_id: int = 1,
    ) -> List[ImportantDate]:
        """Create multiple dates from AI result (parsing will be done by AI processor)."""
        # This is a placeholder - actual date parsing should be done by AI processor
        # For now, we'll store the date text and let the frontend handle parsing
        created_dates = []
        for date_text in dates:
            if date_text and date_text.strip():
                # Try to parse the date (simplified)
                try:
                    # This is a placeholder - actual parsing should be more sophisticated
                    parsed_date = datetime.utcnow()  # Placeholder
                    important_date = ImportantDateService.create_date(
                        db=db,
                        note_id=note_id,
                        date=parsed_date,
                        date_text=date_text.strip(),
                        user_id=user_id,
                    )
                    created_dates.append(important_date)
                except Exception:
                    # If parsing fails, skip this date
                    continue
        
        return created_dates
    
    @staticmethod
    def get_dates_by_note(db: Session, note_id: int) -> List[ImportantDate]:
        """Get all important dates for a specific note."""
        return db.query(ImportantDate).filter(ImportantDate.note_id == note_id).all()
    
    @staticmethod
    def get_upcoming_dates(
        db: Session,
        user_id: int = 1,
        days: int = 30,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ImportantDate]:
        """Get upcoming important dates."""
        today = datetime.utcnow()
        future_date = today + timedelta(days=days)
        
        return db.query(ImportantDate).filter(
            and_(
                ImportantDate.user_id == user_id,
                ImportantDate.date >= today,
                ImportantDate.date <= future_date,
            )
        ).order_by(ImportantDate.date).offset(skip).limit(limit).all()

