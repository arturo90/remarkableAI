"""API endpoints for tasks management."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.base import get_db
from app.db.services import TaskService
from app.db.models import TaskStatus, TaskPriority

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

@router.get("/", response_model=dict)
async def get_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    priority: Optional[str] = None,
    include_duplicates: bool = Query(False),
    db: Session = Depends(get_db),
):
    """Get all tasks with filters."""
    try:
        # Convert status and priority strings to enums
        task_status = None
        if status:
            try:
                task_status = TaskStatus(status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        task_priority = None
        if priority:
            try:
                task_priority = TaskPriority(priority.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
        
        tasks = TaskService.get_all_tasks(
            db=db,
            user_id=1,  # Default user for now
            skip=skip,
            limit=limit,
            status=task_status,
            priority=task_priority,
            include_duplicates=include_duplicates,
        )
        
        # Get statistics
        stats = TaskService.get_task_statistics(db=db, user_id=1)
        
        return {
            "tasks": [
                {
                    "id": task.id,
                    "note_id": task.note_id,
                    "title": task.title,
                    "description": task.description,
                    "status": task.status,
                    "priority": task.priority,
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "category": task.category,
                    "tags": task.tags,
                    "confidence": task.confidence,
                    "is_duplicate": task.is_duplicate,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "updated_at": task.updated_at.isoformat() if task.updated_at else None,
                    "note": {
                        "id": task.note.id,
                        "subject": task.note.subject,
                        "filename": task.note.filename,
                    } if task.note else None,
                }
                for task in tasks
            ],
            "statistics": stats,
            "total": len(tasks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics", response_model=dict)
async def get_task_statistics(
    db: Session = Depends(get_db),
):
    """Get task statistics."""
    try:
        stats = TaskService.get_task_statistics(db=db, user_id=1)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{task_id}", response_model=dict)
async def get_task(
    task_id: int,
    db: Session = Depends(get_db),
):
    """Get a specific task by ID."""
    try:
        task = TaskService.get_task_by_id(db=db, task_id=task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "id": task.id,
            "note_id": task.note_id,
            "title": task.title,
            "description": task.description,
            "status": task.status,
            "priority": task.priority,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "category": task.category,
            "tags": task.tags,
            "source_text": task.source_text,
            "confidence": task.confidence,
            "is_duplicate": task.is_duplicate,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "note": {
                "id": task.note.id,
                "subject": task.note.subject,
                "filename": task.note.filename,
            } if task.note else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{task_id}/status")
async def update_task_status(
    task_id: int,
    status: str,
    db: Session = Depends(get_db),
):
    """Update task status."""
    try:
        task_status = TaskStatus(status.lower())
        task = TaskService.update_task_status(db=db, task_id=task_id, status=task_status)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {
            "id": task.id,
            "status": task.status,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{task_id}/priority")
async def update_task_priority(
    task_id: int,
    priority: str,
    db: Session = Depends(get_db),
):
    """Update task priority."""
    try:
        task_priority = TaskPriority(priority.lower())
        task = TaskService.update_task_priority(db=db, task_id=task_id, priority=task_priority)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {
            "id": task.id,
            "priority": task.priority,
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/{query}", response_model=List[dict])
async def search_tasks(
    query: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Search tasks by text."""
    try:
        tasks = TaskService.search_tasks(
            db=db,
            query=query,
            user_id=1,  # Default user for now
            skip=skip,
            limit=limit,
        )
        return [
            {
                "id": task.id,
                "note_id": task.note_id,
                "title": task.title,
                "status": task.status,
                "priority": task.priority,
                "due_date": task.due_date.isoformat() if task.due_date else None,
                "created_at": task.created_at.isoformat() if task.created_at else None,
            }
            for task in tasks
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{task_id}")
async def delete_task(
    task_id: int,
    db: Session = Depends(get_db),
):
    """Delete a task."""
    try:
        success = TaskService.delete_task(db=db, task_id=task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"message": "Task deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

