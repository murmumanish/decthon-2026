# Docker Commands and Setup Guide

## Docker Build
Command: `docker-compose up --build`


## Docker Stop
Command: `docker-compose down` or `CTRL + C`

## Docker Ollama Models

1. **Granite 4**  
   Link: [https://ollama.com/library/granite4](https://ollama.com/library/granite4)  
   Command: `docker-compose exec ollama ollama pull granite4:350m`


2. **nomic-embed-text**  
Link: [https://ollama.com/library/nomic-embed-text](https://ollama.com/library/nomic-embed-text)  
Command: `docker-compose exec ollama ollama pull nomic-embed-text:latest`


## Listing Ollama Models
Command: `docker-compose exec ollama ollama list`


## Links
- **UI Link**: [http://localhost:3000/](http://localhost:3000/)  
- **Qdrant Dashboard**: [http://localhost:6333/dashboard/](http://localhost:6333/dashboard/)  
- **Ollama**: [http://localhost:11434/](http://localhost:11434/)

## API Testing

### Ask
#### Curl Command: 
```curl
curl -N -X POST "http://localhost:3000/ask/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the town’s transformation using only concrete events from the story."
  }'
```

### Response: 
```
data: hello 
data: my     
    .     
    .     
   
data: John
data: .
```

### Ingest Single Text
#### Curl Command: 
```curl
curl -X POST "http://localhost:3000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a sample document to ingest."
  }'
```

### Response:
```json
{"status":"ok","ingested":1}
```