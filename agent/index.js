import express from "express";
import fetch from "node-fetch";
import { QdrantClient } from "@qdrant/js-client-rest";

const app = express();
app.use(express.json());

const OLLAMA = process.env.OLLAMA_URL; // e.g. http://ollama:11434
const qdrant = new QdrantClient({ url: process.env.QDRANT_URL });

// Create collection once
await qdrant
  .createCollection("docs", {
    vectors: { size: 768, distance: "Cosine" }
  })
  .catch(() => {});

/* ---------- Embeddings ---------- */
async function embed(text) {
  const res = await fetch(`${OLLAMA}/api/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "nomic-embed-text",
      prompt: text
    })
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Ollama embedding failed: ${err}`);
  }

  const json = await res.json();

  if (!Array.isArray(json.embedding)) {
    throw new Error("Invalid embedding response from Ollama");
  }

  return json.embedding;
}

/* ---------- Ingest ---------- */
app.post("/ingest", async (req, res) => {
  const { text, id } = req.body;

  const vector = await embed(text);

  if (vector.length !== 768) {
    throw new Error("Embedding size mismatch");
  }

  await qdrant.upsert("docs", {
    points: [
      {
        id: id ?? Date.now(),
        vector,
        payload: { text }
      }
    ]
  });

  res.json({ status: "stored" });
});

/* ---------- Ask ---------- */
app.post("/ask", async (req, res) => {
  const { question } = req.body;

  const vector = await embed(question);

  const results = await qdrant.search("docs", {
    vector,
    limit: 3
  });

  const context = results
    .map(r => r.payload?.text)
    .filter(Boolean)
    .join("\n");

  const response = await fetch(`${OLLAMA}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "llama3",
      prompt: `Answer using context only:\n${context}\n\nQuestion: ${question}`
    })
  });

  let answer = "";
  let buffer = "";

  for await (const chunk of response.body) {
    buffer += chunk.toString();
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.trim()) continue;
      const json = JSON.parse(line);
      if (json.response) answer += json.response;
    }
  }

  res.json({ answer });
});

/* ---------- Server ---------- */
app.listen(3000, () => {
  console.log("Agent running on http://localhost:3000");
});
