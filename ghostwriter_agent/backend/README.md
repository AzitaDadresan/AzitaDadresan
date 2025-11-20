# GhostWriter Backend API

FastAPI backend server for the GhostWriter multi-agent content system.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
GOOGLE_API_KEY=your_google_key
NANOBANANA_API_KEY=your_nanobanana_key
NANOBANANA_API_URL=https://api.nanobanana.com/v1/generate  # Optional, adjust if needed
PORT=8000  # Optional, defaults to 8000
```

3. Run the server:
```bash
# From project root
python -m backend.main

# Or using uvicorn directly
uvicorn backend.main:app --reload --port 8000
```

## API Endpoints

### WordPress Checker
- `POST /api/check-wordpress` - Check if a URL is a WordPress site

### Image Generation
- `POST /api/generate-image` - Generate an image using nanobanana

### Agent Endpoints
- `POST /api/agents/run-full-cycle` - Run the full GhostWriter agent cycle
- `POST /api/agents/trend-watcher` - Run trend watcher agent
- `POST /api/agents/content-strategist` - Run content strategist agent
- `POST /api/agents/content-creator` - Run content creator agent
- `POST /api/agents/publisher` - Run publisher agent
- `POST /api/agents/evaluator` - Run evaluator agent
- `POST /api/agents/image-generator` - Run image generator agent

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Example Requests

### Check WordPress
```bash
curl -X POST "http://localhost:8000/api/check-wordpress" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

### Generate Image
```bash
curl -X POST "http://localhost:8000/api/generate-image" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A professional business card design", "style": "professional"}'
```

### Run Full Agent Cycle
```bash
curl -X POST "http://localhost:8000/api/agents/run-full-cycle" \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI, career, and women in tech"}'
```

