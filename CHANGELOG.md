# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-08

### Added
- **Database Layer**: Comprehensive database schema with SQLAlchemy
  - User model for multi-user support (future)
  - Note model with metadata, AI results, and processing status
  - Task model with status, priority, due dates, and deduplication
  - Topic model for key themes from notes
  - ImportantDate model for dates and deadlines
- **Database Services**: Complete service layer for database operations
  - NoteService: CRUD operations, search, filtering
  - TaskService: Task management, deduplication, statistics
  - TopicService: Topic management
  - ImportantDateService: Date management
- **RESTful API Endpoints**:
  - `/api/notes/` - List, get, search, delete notes
  - `/api/tasks/` - List, get, update status/priority, search, statistics
- **Database Migrations**: Alembic integration for schema management
- **Database Integration**: Automatic database saving for Gmail sync and uploads
- **Migration Scripts**: Script to migrate file-based data to database
- **Product Plan**: Comprehensive product plan for Apple App of the Year
- **Task Breakdown**: Detailed task breakdown by category (UX, UI, Design, Features)

### Changed
- **API Version**: Updated to v2.0.0
- **Data Persistence**: Notes and tasks now stored in database instead of files only
- **Gmail Service**: Integrated with database for automatic data persistence
- **Main App**: Added database initialization on startup

### Deprecated
- File-based storage for metadata (still works as fallback)
- Old API endpoints for results (new `/api/notes` and `/api/tasks` endpoints available)

### Security
- Database connection properly configured
- User authentication prepared (not yet activated)

### Migration Guide
1. Run database initialization: `python app/db/init_db.py`
2. Run migrations: `alembic upgrade head`
3. Migrate existing data: `PYTHONPATH=. python scripts/migrate_file_data_to_db.py`

## [1.5.0] - Previous Version

### Added
- Manual PDF upload feature with modern UI
- Processing options for uploaded PDFs

## [1.4.0] - Previous Version

### Added
- Enhanced AI system prompts
- PDF viewing functionality
- Improved transcription field

### Changed
- Enhanced OpenAI system prompt
- Improved local model with better rule-based analysis
- Enhanced multimodal LLM prompt

## [1.3.0] - Previous Version

### Added
- Progress bar for AI processing
- Batch processing with individual file tracking

### Fixed
- PDF name/date display

## [1.2.0] - Previous Version

### Fixed
- Buffer API errors
- Frontend data handling
- Local PDF viewing

### Added
- Multimodal AI processing workflow

---

[2.0.0]: https://github.com/arturo90/remarkableAI/releases/tag/v2.0.0
[1.5.0]: https://github.com/arturo90/remarkableAI/releases/tag/v1.5.0
[1.4.0]: https://github.com/arturo90/remarkableAI/releases/tag/v1.4.0
[1.3.0]: https://github.com/arturo90/remarkableAI/releases/tag/v1.3.0
[1.2.0]: https://github.com/arturo90/remarkableAI/releases/tag/v1.2.0

