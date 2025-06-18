# AI Configuration Guide

## Overview
The RemarkableAI application supports both OpenAI and local AI processing for PDF analysis. You can configure which provider to use via environment variables.

## Configuration Options

### Environment Variables

Add these to your `.env` file:

```bash
# AI Processing Configuration
AI_PROVIDER=openai  # "openai" or "local"
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo
```

### AI Providers

#### 1. OpenAI (Recommended for best results)
- **Provider**: `openai`
- **Requirements**: OpenAI API key
- **Features**: 
  - High-quality text analysis
  - Structured JSON responses
  - Task extraction and summarization
  - Date and topic identification

#### 2. Local Processing (Free, basic functionality)
- **Provider**: `local`
- **Requirements**: None (works offline)
- **Features**:
  - Basic keyword-based task extraction
  - Simple text summarization
  - Date pattern matching
  - Rule-based analysis

## Usage Examples

### Using OpenAI
```bash
# Set in .env file
AI_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key-here
```

### Using Local Processing
```bash
# Set in .env file
AI_PROVIDER=local
# No API key needed
```

## API Endpoints

### Process PDF with AI
```bash
POST /gmail/process-with-ai/{message_id}/{attachment_id}
```

### Download and Store PDF
```bash
POST /gmail/download-and-store/{message_id}/{attachment_id}
```

### List Stored PDFs
```bash
GET /gmail/stored-pdfs
```

## Future Enhancements

### Local AI Models
For even better local processing, you can integrate:

1. **Ollama** - Run local LLMs like Llama2, Mistral
2. **Hugging Face** - Use local transformers models
3. **Custom models** - Fine-tuned models for specific use cases

### Example Ollama Integration
```python
# In ai_processor.py
def process_with_ollama(self, text: str) -> Dict[str, Any]:
    import requests
    
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'llama2',
        'prompt': f'Analyze this text: {text}',
        'stream': False
    })
    
    return self.parse_ollama_response(response.json())
```

## Cost Considerations

- **OpenAI**: ~$0.002 per 1K tokens (very affordable for most use cases)
- **Local**: Free but requires more computational resources
- **Hybrid**: Use local for basic processing, OpenAI for complex analysis 