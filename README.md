# RemarkableAI v1.1.0

An intelligent note processing system that automatically analyzes and organizes your Remarkable tablet notes with a comprehensive web interface.

## 🚀 Features

### Core Functionality
- **Automated Gmail Integration**: Fetch PDF notes directly from your Gmail
- **AI-Powered Analysis**: Extract tasks, summaries, dates, and topics from handwritten notes
- **OCR Processing**: Convert handwritten text to searchable content
- **Local & Cloud AI**: Support for both local rule-based and OpenAI processing

### Web Interface (v1.1.0)
- **Enhanced Dashboard**: Real-time statistics and quick actions
- **PDF Management**: Comprehensive PDF viewer with filtering and bulk operations
- **Results Viewer**: Tabbed interface for tasks, summaries, dates, and topics
- **Settings Panel**: Configure AI providers, Gmail filters, and processing options
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices
- **Real-time Updates**: Live status tracking and notifications

### Technical Features
- **FastAPI Backend**: High-performance API with automatic documentation
- **Secure Authentication**: Gmail OAuth integration
- **Local Storage**: Secure local file management
- **Error Handling**: Comprehensive error management and user feedback
- **Extensible Architecture**: Easy to add new AI providers and features

## 📋 Requirements

- Python 3.9+
- Gmail account with API access
- Tesseract OCR (for handwritten text processing)
- OpenAI API key (optional, for advanced AI processing)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/arturo90/remarkableAI.git
cd remarkableAI
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install system dependencies** (macOS):
```bash
brew install tesseract
brew install poppler
```

5. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration:
# - GMAIL_CLIENT_ID
# - GMAIL_CLIENT_SECRET
# - OPENAI_API_KEY (optional)
# - AI_PROVIDER=local (or openai)
```

6. **Start the application**:
```bash
uvicorn app.main:app --reload
```

7. **Access the web interface**:
   - Open http://localhost:8000 in your browser
   - Navigate through the dashboard, PDFs, results, and settings

## 🎯 Quick Start

1. **Access the Dashboard**: Visit http://localhost:8000
2. **Connect Gmail**: Click "Sync from Gmail" to fetch your PDF notes
3. **Process Documents**: Use "Process All" to analyze your notes with AI
4. **View Results**: Check the Results tab to see extracted tasks and summaries
5. **Configure Settings**: Adjust AI providers and processing options

## 📁 Project Structure

```
RemarkableAI/
├── app/
│   ├── api/            # API endpoints (Gmail, processing)
│   ├── core/           # Core configuration and settings
│   ├── services/       # Business logic (Gmail, AI, PDF processing)
│   ├── templates/      # Web interface templates
│   └── utils/          # Utility functions
├── storage/            # Local file storage
│   ├── pdfs/          # Downloaded PDF files
│   └── results/       # AI analysis results
├── tests/              # Test suite
├── config/             # Configuration files
└── docs/               # Documentation
```

## 🔧 Configuration

### Environment Variables
```bash
# Gmail API Configuration
GMAIL_CLIENT_ID=your_gmail_client_id
GMAIL_CLIENT_SECRET=your_gmail_client_secret

# AI Configuration
AI_PROVIDER=local  # or openai
OPENAI_API_KEY=your_openai_api_key  # optional

# Processing Configuration
OCR_ENABLED=true
AUTO_SYNC=false
AUTO_PROCESS=false
```

### Gmail API Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Gmail API
4. Create OAuth 2.0 credentials
5. Add your email as a test user
6. Download credentials and add to `.env` file

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app tests/
```

## 📊 API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## 🔄 Version History

### v1.1.0 (Current)
- Complete UI overhaul with comprehensive web interface
- Enhanced dashboard with real-time statistics
- PDF management with filtering and bulk operations
- Results viewer with tabbed interface
- Settings panel for configuration
- Mobile-responsive design
- Real-time notifications and status tracking

### v1.0.0
- Initial release with core functionality
- Gmail integration and PDF processing
- Basic AI analysis capabilities
- Local storage system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/arturo90/remarkableAI/issues)
- **Documentation**: Check the `/docs` folder for detailed guides
- **API Reference**: Visit http://localhost:8000/docs when running

## 🔮 Roadmap

- [ ] Database integration for persistent storage
- [ ] Advanced AI models (GPT-4, Claude)
- [ ] Team collaboration features
- [ ] Mobile app development
- [ ] Advanced analytics and reporting
- [ ] Integration with task management tools (Todoist, Notion)

---

**Made with ❤️ for Remarkable tablet users** 