"""Test script for database layer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.db.base import SessionLocal
from app.db.models import User
from app.db.services import (
    NoteService, TaskService, TopicService, ImportantDateService
)
from datetime import datetime
import json

def test_database_operations():
    """Test database operations."""
    print("=== Testing Database Layer ===\n")
    
    db = SessionLocal()
    try:
        # Get default user
        user = db.query(User).filter(User.email == "default@remarkableai.com").first()
        if not user:
            print("❌ Default user not found")
            return False
        print(f"✅ Default user found: {user.email} (ID: {user.id})\n")
        
        # Test 1: Create a note
        print("1. Testing note creation...")
        note = NoteService.create_note(
            db=db,
            message_id="test_message_001",
            attachment_id="test_attachment_001",
            subject="Test Note - Database Testing",
            filename="test_note.pdf",
            file_path="storage/pdfs/test_note.pdf",
            from_email="test@remarkable.com",
            received_at=datetime.utcnow(),
            file_size=1024,
            file_hash="test_hash_001",
            user_id=user.id,
        )
        print(f"   ✅ Note created: ID={note.id}, Subject='{note.subject}'")
        
        # Test 2: Update note with AI results
        print("\n2. Testing note update with AI results...")
        ai_result = {
            "summary": "This is a test note about database testing",
            "transcription": "Test transcription of handwritten notes",
            "raw_text": "Raw OCR text from the note",
            "tasks": [
                "Test task 1: Verify database works",
                "Test task 2: Check API endpoints",
                "Test task 3: Validate task creation"
            ],
            "topics": ["Database Testing", "API Development"],
            "dates": ["2025-01-15", "next week"],
            "confidence": 0.95,
            "method": "test"
        }
        
        updated_note = NoteService.update_note_processing_result(
            db=db,
            note_id=note.id,
            summary=ai_result["summary"],
            transcription=ai_result["transcription"],
            raw_text=ai_result["raw_text"],
            confidence=ai_result["confidence"],
            ai_result_json=ai_result,
            processing_method=ai_result["method"],
            processing_status="completed",
        )
        print(f"   ✅ Note updated: Status={updated_note.processing_status}, Confidence={updated_note.confidence}")
        
        # Test 3: Create tasks from AI result
        print("\n3. Testing task creation...")
        tasks = TaskService.create_tasks_from_ai_result(
            db=db,
            note_id=note.id,
            tasks=ai_result["tasks"],
            user_id=user.id,
        )
        print(f"   ✅ Created {len(tasks)} tasks")
        for task in tasks:
            print(f"      - {task.title} (Status: {task.status}, Priority: {task.priority})")
        
        # Test 4: Create topics from AI result
        print("\n4. Testing topic creation...")
        topics = TopicService.create_topics_from_ai_result(
            db=db,
            note_id=note.id,
            topics=ai_result["topics"],
            user_id=user.id,
        )
        print(f"   ✅ Created {len(topics)} topics")
        for topic in topics:
            print(f"      - {topic.name}")
        
        # Test 5: Create dates from AI result
        print("\n5. Testing date creation...")
        dates = ImportantDateService.create_dates_from_ai_result(
            db=db,
            note_id=note.id,
            dates=ai_result["dates"],
            user_id=user.id,
        )
        print(f"   ✅ Created {len(dates)} dates")
        for date in dates:
            print(f"      - {date.date_text} (Date: {date.date})")
        
        # Test 6: Get all notes
        print("\n6. Testing note retrieval...")
        all_notes = NoteService.get_notes(db=db, user_id=user.id, limit=10)
        print(f"   ✅ Retrieved {len(all_notes)} notes")
        
        # Test 7: Get all tasks
        print("\n7. Testing task retrieval...")
        all_tasks = TaskService.get_all_tasks(db=db, user_id=user.id, limit=10)
        print(f"   ✅ Retrieved {len(all_tasks)} tasks")
        
        # Test 8: Get task statistics
        print("\n8. Testing task statistics...")
        stats = TaskService.get_task_statistics(db=db, user_id=user.id)
        print(f"   ✅ Task Statistics:")
        print(f"      - Total: {stats['total']}")
        print(f"      - Pending: {stats['pending']}")
        print(f"      - Completed: {stats['completed']}")
        print(f"      - Overdue: {stats['overdue']}")
        print(f"      - Completion Rate: {stats['completion_rate']:.1f}%")
        
        # Test 9: Update task status
        print("\n9. Testing task status update...")
        if all_tasks:
            task = all_tasks[0]
            from app.db.models import TaskStatus
            updated_task = TaskService.update_task_status(
                db=db,
                task_id=task.id,
                status=TaskStatus.IN_PROGRESS,
            )
            print(f"   ✅ Task status updated: {updated_task.status}")
        
        # Test 10: Search notes
        print("\n10. Testing note search...")
        search_results = NoteService.search_notes(
            db=db,
            query="database",
            user_id=user.id,
            limit=10,
        )
        print(f"   ✅ Found {len(search_results)} notes matching 'database'")
        
        # Test 11: Search tasks
        print("\n11. Testing task search...")
        task_search_results = TaskService.search_tasks(
            db=db,
            query="test",
            user_id=user.id,
            limit=10,
        )
        print(f"   ✅ Found {len(task_search_results)} tasks matching 'test'")
        
        print("\n" + "=" * 50)
        print("✅ All database tests passed!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = test_database_operations()
    sys.exit(0 if success else 1)

