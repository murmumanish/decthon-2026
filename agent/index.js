import express from "express";
import path from 'path';
import { fileURLToPath } from 'url';
import fetch from "node-fetch";
import { QdrantClient } from "@qdrant/js-client-rest";


const app = express();
const PORT = 3000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


// ---------- Middleware ----------
app.use(express.json());

app.use(express.static(path.resolve(__dirname, 'ui')));

// ---------- Qdrant + Ollama setup ----------
const OLLAMA = process.env.OLLAMA_URL; 
const qdrant = new QdrantClient({ url: process.env.QDRANT_URL });

// Create collection if it doesn't exist
await qdrant
  .createCollection("docs", {
    vectors: { size: 768, distance: "Cosine" }
  })
  .catch(() => { console.log("Collection 'docs' already exists"); });

/* ---------- Embedding function ---------- */
async function embed(text) {
  const res = await fetch(`${OLLAMA}/api/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "nomic-embed-text", prompt: text })
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

// Serving the HTML
app.get('/',(req,res) => {
    res.sendFile(path.resolve(__dirname, 'ui', 'index.html'));
  })


/* ---------- Ingest endpoint ---------- */
app.post("/ingest", async (req, res) => {
  try {
    const { text, id } = req.body;

    if (!text || text.trim() === "") {
      return res.status(400).json({ error: "Text is required" });
    }

    const vector = await embed(text);

    if (vector.length !== 768) {
      throw new Error("Embedding size mismatch");
    }

    // Ensure valid Qdrant ID
    let pointId;
    if (id && !isNaN(Number(id))) {
      pointId = Number(id);
    } else {
      pointId = Date.now(); // fallback
    }

    await qdrant.upsert("docs", {
      points: [
        { id: pointId, vector, payload: { text } }
      ]
    });

    res.json({ status: "stored", id: pointId });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

/* ---------- Ask endpoint ---------- */
app.post("/ask", async (req, res) => {
  try {
    const { question } = req.body;
    if (!question || question.trim() === "") {
      return res.status(400).json({ error: "Question is required" });
    }

    const vector = await embed(question);

    const results = await qdrant.search("docs", { vector, limit: 10 });

    const context = results
      .map(r => r.payload?.text)
      .filter(Boolean)
      .join("\n");

    const response = await fetch(`${OLLAMA}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "granite4:350m",
        //prompt: `Answer using context only:\n${context}\n\nQuestion: ${question}`
        prompt:`You are an AI assistant using retrieval-augmented generation.Answer \n\nQuestion: ${question} ONLY using the provided \n${context}.If the answer is not present in the \n${context}, respond with:"I don't have enough information to answer that."
`
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
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

/* ---------- Start server ---------- */
app.listen(PORT, () => {
  console.log(`Agent running on http://localhost:${PORT}`);
});
