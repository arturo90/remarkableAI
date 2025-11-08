"""Migration script to migrate file-based data to database."""
import sys
import json
from pathlib import Path
from datetime import datetime
import hashlib

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.base import SessionLocal
from app.db.models import User
from app.db.services import (
    NoteService, TaskService, TopicService, ImportantDateService
)

def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime."""
    try:
        # Try parsing as milliseconds timestamp
        timestamp = int(timestamp_str) / 1000
        return datetime.fromtimestamp(timestamp)
    except (ValueError, TypeError):
        # Try parsing as ISO format
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            # Fallback to current time
            return datetime.utcnow()

def migrate_pdfs_to_notes(db: SessionLocal):
    """Migrate PDF files and metadata to notes table."""
    pdfs_dir = Path("storage/pdfs")
    if not pdfs_dir.exists():
        print("No PDFs directory found. Skipping PDF migration.")
        return 0
    
    migrated_count = 0
    skipped_count = 0
    
    # Get default user
    default_user = db.query(User).filter(User.email == "default@remarkableai.com").first()
    if not default_user:
        print("Error: Default user not found. Please run init_db.py first.")
        return 0
    
    # Iterate through PDF files
    for pdf_file in pdfs_dir.glob("*.pdf"):
        try:
            # Check if corresponding JSON metadata exists
            metadata_file = pdf_file.with_suffix('.json')
            if not metadata_file.exists():
                print(f"Skipping {pdf_file.name}: No metadata file found")
                skipped_count += 1
                continue
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if note already exists
            message_id = metadata.get('message_id', f"migrated_{pdf_file.stem}")
            existing_note = NoteService.get_note_by_message_id(db, message_id)
            if existing_note:
                print(f"Skipping {pdf_file.name}: Note already exists in database")
                skipped_count += 1
                continue
            
            # Calculate file hash
            file_hash = get_file_hash(pdf_file)
            
            # Check for duplicate by hash
            existing_note_by_hash = NoteService.get_note_by_hash(db, file_hash)
            if existing_note_by_hash:
                print(f"Skipping {pdf_file.name}: Duplicate file (same hash)")
                skipped_count += 1
                continue
            
            # Parse received_at timestamp
            received_at = None
            if 'date' in metadata:
                received_at = parse_timestamp(str(metadata['date']))
            elif 'received_at' in metadata:
                received_at = parse_timestamp(str(metadata['received_at']))
            
            # Create note
            note = NoteService.create_note(
                db=db,
                message_id=message_id,
                attachment_id=metadata.get('attachment_id', f"migrated_{pdf_file.stem}"),
                subject=metadata.get('subject', pdf_file.stem),
                filename=metadata.get('filename', pdf_file.name),
                file_path=str(pdf_file),
                from_email=metadata.get('from', 'Unknown'),
                received_at=received_at,
                file_size=pdf_file.stat().st_size,
                file_hash=file_hash,
                user_id=default_user.id,
            )
            
            print(f"Migrated note: {note.subject} (ID: {note.id})")
            migrated_count += 1
            
        except Exception as e:
            print(f"Error migrating {pdf_file.name}: {str(e)}")
            skipped_count += 1
            continue
    
    return migrated_count, skipped_count

def migrate_ai_results_to_notes(db: SessionLocal):
    """Migrate AI results to notes."""
    results_dir = Path("storage/results")
    if not results_dir.exists():
        print("No results directory found. Skipping AI results migration.")
        return 0, 0
    
    migrated_count = 0
    skipped_count = 0
    
    # Iterate through result files
    for result_file in results_dir.glob("*.json"):
        try:
            # Load result data
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            # Get message_id and attachment_id
            message_id = result_data.get('message_id')
            attachment_id = result_data.get('attachment_id')
            
            if not message_id or not attachment_id:
                print(f"Skipping {result_file.name}: Missing message_id or attachment_id")
                skipped_count += 1
                continue
            
            # Find note
            note = NoteService.get_note_by_message_id(db, message_id)
            if not note:
                print(f"Skipping {result_file.name}: Note not found for message_id {message_id}")
                skipped_count += 1
                continue
            
            # Check if note already has processing results
            if note.processing_status == "completed" and note.summary:
                print(f"Skipping {result_file.name}: Note already has processing results")
                skipped_count += 1
                continue
            
            # Extract AI result
            ai_result = result_data.get('result', {})
            
            # Update note with AI results
            NoteService.update_note_processing_result(
                db=db,
                note_id=note.id,
                summary=ai_result.get('summary'),
                transcription=ai_result.get('transcription'),
                raw_text=ai_result.get('raw_text', ''),
                confidence=ai_result.get('confidence', 0.0),
                ai_result_json=ai_result,
                processing_method=ai_result.get('method', 'unknown'),
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
            
            print(f"Migrated AI results for note: {note.subject} (ID: {note.id})")
            migrated_count += 1
            
        except Exception as e:
            print(f"Error migrating {result_file.name}: {str(e)}")
            skipped_count += 1
            continue
    
    return migrated_count, skipped_count

def main():
    """Main migration function."""
    print("Starting migration from file-based storage to database...")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        # Migrate PDFs to notes
        print("\n1. Migrating PDFs to notes...")
        pdf_migrated, pdf_skipped = migrate_pdfs_to_notes(db)
        print(f"   Migrated: {pdf_migrated}, Skipped: {pdf_skipped}")
        
        # Migrate AI results
        print("\n2. Migrating AI results to notes...")
        results_migrated, results_skipped = migrate_ai_results_to_notes(db)
        print(f"   Migrated: {results_migrated}, Skipped: {results_skipped}")
        
        print("\n" + "=" * 60)
        print("Migration complete!")
        print(f"Total notes migrated: {pdf_migrated}")
        print(f"Total AI results migrated: {results_migrated}")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    main()

