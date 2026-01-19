import "dotenv/config";

export const ENV = {
  QDRANT_URL: process.env.QDRANT_URL || "http://qdrant:6333",
  OLLAMA_URL: process.env.OLLAMA_URL || "http://ollama:11434",
  COLLECTION: "my_collection",
};
