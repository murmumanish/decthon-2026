import { ChatOllama } from "@langchain/ollama";
import { ENV } from "../config/env.js";

export function createLLM({ streaming = false } = {}) {
  return new ChatOllama({
    model: "granite4:350m",
    temperature: 0,
    streaming,
    baseUrl: ENV.OLLAMA_URL,
  });
}
