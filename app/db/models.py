"""Database models for RemarkableAI."""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime
from app.db.base import Base


class TaskPriority(str, enum.Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(str, enum.Enum):
    """Task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class User(Base):
    """User model for multi-user support (future)."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    notes = relationship("Note", back_populates="user", cascade="all, delete-orphan")


class Note(Base):
    """Note model - represents a processed PDF note."""
    __tablename__ = "notes"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, default=1)  # Default to user 1 for now
    
    # Gmail metadata
    message_id = Column(String(255), unique=True, index=True, nullable=False)
    attachment_id = Column(String(255), index=True, nullable=False)
    subject = Column(String(500), nullable=False)
    from_email = Column(String(255))
    received_at = Column(DateTime(timezone=True))
    
    # File metadata
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer)  # Size in bytes
    file_hash = Column(String(64), index=True)  # MD5 hash for deduplication
    
    # Processing metadata
    processed_at = Column(DateTime(timezone=True))
    processing_method = Column(String(50))  # 'openai', 'local', 'multimodal'
    processing_status = Column(String(50), default="pending")  # 'pending', 'processing', 'completed', 'failed'
    processing_error = Column(Text)
    
    # AI analysis results
    summary = Column(Text)
    transcription = Column(Text)  # Clean, formatted transcription
    raw_text = Column(Text)  # Raw OCR/text extraction
    confidence = Column(Float, default=0.0)  # Confidence score 0-1
    ai_result_json = Column(JSON)  # Full AI result as JSON
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="notes")
    tasks = relationship("Task", back_populates="note", cascade="all, delete-orphan")
    topics = relationship("Topic", back_populates="note", cascade="all, delete-orphan")
    dates = relationship("ImportantDate", back_populates="note", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Note(id={self.id}, subject='{self.subject}', message_id='{self.message_id}')>"


class Task(Base):
    """Task model - represents an actionable task extracted from notes."""
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    note_id = Column(Integer, ForeignKey("notes.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, default=1)
    
    # Task details
    title = Column(String(500), nullable=False, index=True)
    description = Column(Text)
    status = Column(String(50), default=TaskStatus.PENDING.value, index=True)
    priority = Column(String(50), default=TaskPriority.MEDIUM.value, index=True)
    
    # Dates
    due_date = Column(DateTime(timezone=True), index=True)
    completed_at = Column(DateTime(timezone=True))
    
    # Metadata
    category = Column(String(100), index=True)
    tags = Column(JSON)  # List of tags as JSON array
    source_text = Column(Text)  # Original text from note
    confidence = Column(Float, default=0.0)  # Confidence score 0-1
    
    # Deduplication
    task_hash = Column(String(64), index=True)  # Hash for deduplication
    is_duplicate = Column(Boolean, default=False)
    original_task_id = Column(Integer, ForeignKey("tasks.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    note = relationship("Note", back_populates="tasks")
    user = relationship("User")
    original_task = relationship("Task", remote_side=[id])
    
    def __repr__(self):
        return f"<Task(id={self.id}, title='{self.title}', status={self.status.value}, priority={self.priority.value})>"


class Topic(Base):
    """Topic model - represents a topic or theme extracted from notes."""
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True, index=True)
    note_id = Column(Integer, ForeignKey("notes.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, default=1)
    
    # Topic details
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    relevance_score = Column(Float, default=0.0)  # Relevance score 0-1
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    note = relationship("Note", back_populates="topics")
    user = relationship("User")
    
    def __repr__(self):
        return f"<Topic(id={self.id}, name='{self.name}', note_id={self.note_id})>"


class ImportantDate(Base):
    """ImportantDate model - represents important dates extracted from notes."""
    __tablename__ = "important_dates"
    
    id = Column(Integer, primary_key=True, index=True)
    note_id = Column(Integer, ForeignKey("notes.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, default=1)
    
    # Date details
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    date_text = Column(String(255))  # Original text from note (e.g., "next week", "June 15")
    description = Column(Text)  # Context or description
    date_type = Column(String(50))  # 'deadline', 'meeting', 'event', 'reminder', etc.
    is_recurring = Column(Boolean, default=False)
    recurrence_pattern = Column(String(100))  # e.g., "weekly", "monthly"
    
    # Metadata
    confidence = Column(Float, default=0.0)  # Confidence score 0-1
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    note = relationship("Note", back_populates="dates")
    user = relationship("User")
    
    def __repr__(self):
        return f"<ImportantDate(id={self.id}, date={self.date}, note_id={self.note_id})>"


# Indexes for full-text search (SQLite)
# Note: For PostgreSQL, we'll use GIN indexes with tsvector
# For SQLite, we'll use FTS5 virtual tables in a separate migration

