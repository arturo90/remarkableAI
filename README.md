# RemarkableAI

An intelligent note processing system that automatically analyzes and organizes your Remarkable tablet notes.

## Features

- Automated Gmail integration for fetching Remarkable PDF notes
- AI-powered note analysis using OpenAI
- Task extraction and organization
- Daily and weekly summaries
- Master task list management

## Project Structure

```
remarkableai/
├── app/
│   ├── api/            # API endpoints
│   ├── core/           # Core application logic
│   ├── db/             # Database models and migrations
│   ├── services/       # Business logic services
│   └── utils/          # Utility functions
├── tests/              # Test suite
├── docs/               # Documentation
├── scripts/            # Utility scripts
└── config/             # Configuration files
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/remarkableai.git
cd remarkableai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
alembic upgrade head
```

## Development

1. Run tests:
```bash
pytest
```

2. Start the development server:
```bash
uvicorn app.main:app --reload
```

## Testing

The project uses pytest for testing. Run tests with:
```bash
pytest
```

For coverage reports:
```bash
pytest --cov=app tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Security

- All API keys and credentials are stored securely
- Data encryption is implemented
- Regular security audits are performed

## Support

For support, please open an issue in the GitHub repository. 