from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys
import time

sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.retrieval.hybrid_retriever import HybridRetriever
from athena.retrieval.hyde_retriever import HyDERetriever
from athena.retrieval.multi_query_retriever import MultiQueryRetriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator
from athena.pipelines.rag_pipeline import RAGPipeline
from athena.pipelines.crag import CRAGPipeline
from athena.pipelines.adaptive_rag import AdaptiveRAGPipeline
from athena.agents.react_agent import ReActAgent
from athena.agents.planner_executor import PlannerExecutorAgent
from athena.agents.reflection_agent import ReflectionAgent
from athena.agents.multi_agent import MultiAgentOrchestrator

# Global components
components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Athena components...")
    embedder  = BGEEmbedder()
    store     = QdrantStore()
    dense     = DenseRetriever(embedder, store)
    bm25      = BM25Retriever()
    bm25.load()
    hybrid    = HybridRetriever(dense, bm25)
    hyde      = HyDERetriever(embedder, store)
    multi_q   = MultiQueryRetriever(dense)
    reranker  = BGEReranker()
    generator = OllamaGenerator()

    components["pipelines"] = {
        "rag":      RAGPipeline(dense, bm25, reranker, generator),
        "crag":     CRAGPipeline(hybrid, reranker, generator),
        "adaptive": AdaptiveRAGPipeline(dense, bm25, hyde, multi_q,
                                         reranker, generator),
    }
    components["agents"] = {
        "react":      ReActAgent(hybrid, reranker),
        "planner":    PlannerExecutorAgent(hybrid, reranker, generator),
        "reflection": ReflectionAgent(hybrid, reranker),
        "multi":      MultiAgentOrchestrator(hybrid, multi_q,
                                              reranker, generator),
    }
    print("Athena ready.")
    yield
    components.clear()

app = FastAPI(
    title="Athena — AI Research Intelligence API",
    description="Production RAG system over 100 AI/ML research papers.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Request / Response models ──────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    pipeline: str = "rag"  # rag | crag | adaptive

class AgentRequest(BaseModel):
    question: str
    agent: str = "react"  # react | planner | reflection | multi

class QueryResponse(BaseModel):
    question:      str
    answer:        str
    sources:       list[str]
    pipeline:      str
    latency_ms:    float

class AgentResponse(BaseModel):
    question:    str
    answer:      str
    sources:     list[str]
    agent:       str
    latency_ms:  float

# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "pipelines": list(components.get("pipelines", {}).keys()),
        "agents":    list(components.get("agents", {}).keys()),
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    pipelines = components.get("pipelines", {})
    if request.pipeline not in pipelines:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pipeline '{request.pipeline}'. "
                   f"Choose from: {list(pipelines.keys())}"
        )
    start  = time.time()
    result = pipelines[request.pipeline].query(request.question)
    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=list(set(result.get("sources", []))),
        pipeline=request.pipeline,
        latency_ms=round((time.time() - start) * 1000, 2),
    )

@app.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest):
    agents = components.get("agents", {})
    if request.agent not in agents:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent '{request.agent}'. "
                   f"Choose from: {list(agents.keys())}"
        )
    start  = time.time()
    result = agents[request.agent].run(request.question)
    return AgentResponse(
        question=request.question,
        answer=result["answer"],
        sources=list(set(result.get("sources", []))),
        agent=request.agent,
        latency_ms=round((time.time() - start) * 1000, 2),
    )

@app.get("/pipelines")
def list_pipelines():
    return {"pipelines": list(components.get("pipelines", {}).keys())}

@app.get("/agents")
def list_agents():
    return {"agents": list(components.get("agents", {}).keys())}