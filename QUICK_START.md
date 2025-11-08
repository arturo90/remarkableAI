# Quick Start Guide - RemarkableAI v2.0.0

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Virtual environment activated
- Database dependencies installed

### Installation

1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize database**:
   ```bash
   PYTHONPATH=. python app/db/init_db.py
   ```

4. **Run database migrations** (if needed):
   ```bash
   alembic upgrade head
   ```

5. **Start the server**:
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Access the application**:
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 📊 Database Status

### Check Database
```bash
# Check database connection
python -c "from app.db.base import SessionLocal; from app.db.models import User; db = SessionLocal(); users = db.query(User).all(); print(f'Users: {len(users)}'); db.close()"
```

### View Database Schema
```bash
# List all tables
sqlite3 remarkable_ai.db ".tables"

# View notes
sqlite3 remarkable_ai.db "SELECT id, subject, processing_status FROM notes;"

# View tasks
sqlite3 remarkable_ai.db "SELECT id, title, status, priority FROM tasks;"
```

## 🧪 Testing

### Test Database Layer
```bash
python test_database.py
```

### Test API Endpoints
```bash
# Start server first
uvicorn app.main:app --reload

# In another terminal, run:
python test_api_endpoints.py
```

### Manual API Testing
```bash
# Health check
curl http://localhost:8000/health

# Get all notes
curl http://localhost:8000/api/notes/

# Get all tasks
curl http://localhost:8000/api/tasks/

# Get task statistics
curl http://localhost:8000/api/tasks/statistics

# Search notes
curl http://localhost:8000/api/notes/search/database

# Search tasks
curl http://localhost:8000/api/tasks/search/test
```

## 📝 Migrating Existing Data

If you have existing file-based data, migrate it to the database:

```bash
PYTHONPATH=. python scripts/migrate_file_data_to_db.py
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with:
```bash
# Database
DATABASE_URL=sqlite:///./remarkable_ai.db

# Gmail API (if using)
GMAIL_CLIENT_ID=your_client_id
GMAIL_CLIENT_SECRET=your_client_secret

# AI Processing
AI_PROVIDER=local  # Options: local, openai, multimodal
OPENAI_API_KEY=your_openai_key  # If using OpenAI
```

## 🎯 Key Features

### Database Layer
- ✅ SQLAlchemy ORM with SQLite (dev) / PostgreSQL (production)
- ✅ Alembic migrations
- ✅ User, Note, Task, Topic, ImportantDate models
- ✅ Automatic data persistence

### API Endpoints
- ✅ `/api/notes/` - Notes management
- ✅ `/api/tasks/` - Task management
- ✅ Search and filtering
- ✅ Statistics and analytics

### Task Management
- ✅ Task creation from AI results
- ✅ Task deduplication
- ✅ Status and priority management
- ✅ Due dates and completion tracking

## 🐛 Troubleshooting

### Database Issues
```bash
# Reset database (WARNING: Deletes all data)
rm remarkable_ai.db
python app/db/init_db.py
alembic upgrade head
```

### Migration Issues
```bash
# Check current migration
alembic current

# View migration history
alembic history

# Rollback last migration
alembic downgrade -1
```

### Server Issues
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process on port 8000
kill -9 $(lsof -t -i:8000)
```

## 📚 API Documentation

Visit http://localhost:8000/docs for interactive API documentation.

## 🆘 Support

- Check `TEST_RESULTS.md` for test results
- Check `IMPLEMENTATION_STATUS.md` for implementation status
- Check `PRODUCT_PLAN.md` for product roadmap

---

**Version**: 2.0.0  
**Last Updated**: January 2025

