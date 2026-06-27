from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys
import time
from prometheus_client import Counter, Histogram, make_asgi_app, REGISTRY

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
from athena.api.cache import SemanticCache

# ── Prometheus metrics ─────────────────────────────────────────────────
QUERY_COUNT = Counter(
    "athena_queries_total",
    "Total number of queries",
    ["pipeline", "status"]
)
QUERY_LATENCY = Histogram(
    "athena_query_duration_seconds",
    "Query latency in seconds",
    ["pipeline"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)
AGENT_COUNT = Counter(
    "athena_agent_runs_total",
    "Total number of agent runs",
    ["agent", "status"]
)
AGENT_LATENCY = Histogram(
    "athena_agent_duration_seconds",
    "Agent run latency in seconds",
    ["agent"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)
CACHE_HITS = Counter(
    "athena_cache_hits_total",
    "Total semantic cache hits"
)

# Global components
components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Athena components...")
    embedder  = BGEEmbedder()
    cache = SemanticCache(embedder, threshold=0.85)
    components["cache"] = cache
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

metrics_app = make_asgi_app()
app.mount("/metrics/", metrics_app)

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
    cache     = components.get("cache")

    if request.pipeline not in pipelines:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pipeline '{request.pipeline}'. "
                   f"Choose from: {list(pipelines.keys())}"
        )

    start = time.time()
    try:
        # Check semantic cache first
        if cache:
            cached = cache.get(request.question)
            if cached:
                CACHE_HITS.inc()
                return QueryResponse(
                    question=request.question,
                    answer=cached["answer"],
                    sources=cached["sources"],
                    pipeline=f"{request.pipeline}_cached",
                    latency_ms=round((time.time() - start) * 1000, 2),
                )

        result = pipelines[request.pipeline].query(request.question)
        latency = time.time() - start

        # Store in cache
        if cache:
            cache.set(request.question, result["answer"],
                      result.get("sources", []))

        QUERY_COUNT.labels(pipeline=request.pipeline, status="success").inc()
        QUERY_LATENCY.labels(pipeline=request.pipeline).observe(latency)
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=list(set(result.get("sources", []))),
            pipeline=request.pipeline,
            latency_ms=round(latency * 1000, 2),
        )
    except Exception as e:
        QUERY_COUNT.labels(pipeline=request.pipeline, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/cache")
def clear_cache():
    cache = components.get("cache")
    if not cache:
        raise HTTPException(status_code=503, detail="Cache not available")
    cleared = cache.clear()
    return {"cleared": cleared}

@app.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest):
    agents = components.get("agents", {})
    if request.agent not in agents:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent '{request.agent}'. "
                   f"Choose from: {list(agents.keys())}"
        )
    start = time.time()
    try:
        result = agents[request.agent].run(request.question)
        latency = time.time() - start
        AGENT_COUNT.labels(agent=request.agent, status="success").inc()
        AGENT_LATENCY.labels(agent=request.agent).observe(latency)
        return AgentResponse(
            question=request.question,
            answer=result["answer"],
            sources=list(set(result.get("sources", []))),
            agent=request.agent,
            latency_ms=round(latency * 1000, 2),
        )
    except Exception as e:
        AGENT_COUNT.labels(agent=request.agent, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipelines")
def list_pipelines():
    return {"pipelines": list(components.get("pipelines", {}).keys())}

@app.get("/agents")
def list_agents():
    return {"agents": list(components.get("agents", {}).keys())}