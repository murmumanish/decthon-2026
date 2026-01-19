import { RetrievalQAChain } from "@langchain/classic/chains";
import { createLLM } from "../llm/ollama.js";
import { getVectorStore } from "./vectorstore.js";

let chain;

export async function getRagChain({ streaming = false } = {}) {
  if (chain) return chain;

  const llm = createLLM({ streaming });
  const vectorStore = await getVectorStore();
  const retriever = vectorStore.asRetriever(5);

  chain = RetrievalQAChain.fromLLM(llm, retriever);
  return chain;
}

