import express from "express";
import path from 'path';
import { fileURLToPath } from 'url';
import { getRagChain } from "../rag/chain.js";
import { ingestTexts } from "../rag/vectorstore.js";

const app = express();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


app.use(express.json());
app.use(express.static(path.resolve(__dirname, '../ui')));

// Serving UI
app.get('/',(req,res) => {
    res.sendFile(path.resolve(__dirname, '../ui', 'index.html'));
  })


// 🔹 Ingest endpoint
app.post("/ingest", async (req, res) => {
  try {
    const { text, texts } = req.body;

    // Normalize input → always array
    const docs =
      Array.isArray(texts) ? texts :
      typeof text === "string" ? [text] :
      [];

    if (docs.length === 0) {
      return res.status(400).json({
        error: "No text(s) provided for ingestion",
      });
    }

    await ingestTexts(docs);

    res.json({ status: "ok", ingested: docs.length });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});


// 🔹 RAG (non-streaming)
app.post("/ask", async (req, res) => {
  const chain = await getRagChain();
  const result = await chain.call({ query: req.body.question });
  res.json({ answer: result.text });
});

// 🔹 RAG (streaming)
app.post("/ask/stream", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const chain = await getRagChain({ streaming: true });
  /*
  const stream = await chain.stream({ query: req.body.question });

  for await (const chunk of stream) {
    if (chunk.text){
      res.write(chunk.text);
    }
  }*/

  const stream = await chain.streamEvents(
    { query: req.body.question },
    { version: "v1" }
  );

  for await (const event of stream) {
    if (event.event === "on_llm_stream") {
      const token = event.data.chunk?.content;
      if (token) {
        res.write(`data: ${token}\n\n`);
      }
    }
  }


  res.end();
});

export default app;

