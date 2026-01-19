import { OllamaEmbeddings } from "@langchain/ollama";
import { ENV } from "../config/env.js";

export function createEmbeddings() {
  return new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: ENV.OLLAMA_URL,
  });
}

