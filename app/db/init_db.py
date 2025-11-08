"""Initialize database with default user."""
from app.db.base import SessionLocal, init_db, engine
from app.db.models import User
from app.db.services import NoteService, TaskService, TopicService, ImportantDateService

def create_default_user():
    """Create a default user if it doesn't exist."""
    db = SessionLocal()
    try:
        # Check if default user exists
        default_user = db.query(User).filter(User.email == "default@remarkableai.com").first()
        
        if not default_user:
            default_user = User(
                email="default@remarkableai.com",
                username="default",
                is_active=True,
            )
            db.add(default_user)
            db.commit()
            db.refresh(default_user)
            print(f"Created default user with ID: {default_user.id}")
        else:
            print(f"Default user already exists with ID: {default_user.id}")
        
        return default_user
    finally:
        db.close()

def initialize_database():
    """Initialize the database."""
    print("Initializing database...")
    init_db()
    print("Database initialized.")
    
    print("Creating default user...")
    create_default_user()
    print("Database setup complete!")

if __name__ == "__main__":
    initialize_database()

