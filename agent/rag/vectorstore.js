import { QdrantVectorStore } from "@langchain/qdrant";
import { createEmbeddings } from "./embeddings.js";
import { ENV } from "../config/env.js";

let vectorStore;

export async function getVectorStore() {
  if (vectorStore) return vectorStore;

  const embeddings = createEmbeddings();

  vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: ENV.QDRANT_URL,
      collectionName: ENV.COLLECTION,
    }
  );

  return vectorStore;
}

export async function ingestTexts(texts) {
  const store = await getVectorStore();

  await store.addDocuments(
    texts.map((text, i) => ({
      pageContent: text,
      metadata: { id: i + 1 },
    }))
  );
}

