import sys
import os
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
import json
from pathlib import Path

sys.path.insert(0, "src")

from athena.ingestion.embedders.bge import BGEEmbedder
from athena.retrieval.qdrant_store import QdrantStore
from athena.retrieval.dense_retriever import DenseRetriever
from athena.retrieval.bm25_retriever import BM25Retriever
from athena.reranking.bge_reranker import BGEReranker
from athena.pipelines.generator import OllamaGenerator
from athena.pipelines.rag_pipeline import RAGPipeline
import httpx

def llm_judge(question: str, answer: str, ground_truth: str,
              context: list[str], model: str = OLLAMA_MODEL) -> dict:
    context_str = "\n\n".join(context[:3])
    prompt = f"""You are an expert evaluator for RAG systems. Evaluate the answer strictly.

Question: {question}
Ground Truth: {ground_truth}
Retrieved Context: {context_str}
Generated Answer: {answer}

Score each dimension from 0.0 to 1.0:
1. faithfulness: Are all claims in the answer supported by the context?
2. answer_relevancy: Does the answer address the question?
3. correctness: Does the answer align with the ground truth?

Respond in valid JSON only:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "correctness": 0.0, "reasoning": ""}}"""

    resp = httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp_json = resp.json()
    if "response" not in resp_json:
        print(f"  ⚠ Ollama error: {resp_json}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "correctness": 0.0, "reasoning": "ollama_error"}
    text = resp_json["response"].strip()
    try:
        start = text.index("{")
        end   = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "correctness": 0.0, "reasoning": "parse error"}


def run_evaluation(pipeline: RAGPipeline, questions_path: str,
                   results_path: str, run_name: str):
    questions = json.loads(Path(questions_path).read_text())
    results   = []

    print(f"\nRunning evaluation: {run_name}")
    print(f"Questions: {len(questions)}\n")

    for i, item in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {item['question'][:60]}...")
        try:
            output = pipeline.query(item["question"])
        except Exception as e:
            print(f"     ⚠ Pipeline error: {e}")
            results.append({
                "question":     item["question"],
                "ground_truth": item["ground_truth"],
                "answer":       "",
                "sources":      [],
                "scores":       {"faithfulness": 0.0,
                                 "answer_relevancy": 0.0,
                                 "correctness": 0.0,
                                 "reasoning": str(e)},
            })
            continue
        scores = llm_judge(
            question=item["question"],
            answer=output["answer"],
            ground_truth=item["ground_truth"],
            context=output["chunks"],
        )
        results.append({
            "question":     item["question"],
            "ground_truth": item["ground_truth"],
            "answer":       output["answer"],
            "sources":      output["sources"],
            "scores":       scores,
        })
        print(f"     faithfulness={scores.get('faithfulness', 0):.2f} "
              f"relevancy={scores.get('answer_relevancy', 0):.2f} "
              f"correctness={scores.get('correctness', 0):.2f}")

    avg_faithfulness  = sum(r["scores"].get("faithfulness", 0)  for r in results) / len(results)
    avg_relevancy     = sum(r["scores"].get("answer_relevancy", 0) for r in results) / len(results)
    avg_correctness   = sum(r["scores"].get("correctness", 0)   for r in results) / len(results)

    summary = {
        "run_name":        run_name,
        "num_questions":   len(questions),
        "avg_faithfulness":  round(avg_faithfulness, 4),
        "avg_relevancy":     round(avg_relevancy, 4),
        "avg_correctness":   round(avg_correctness, 4),
        "results":         results,
    }

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*50}")
    print(f"Run: {run_name}")
    print(f"  Faithfulness:  {avg_faithfulness:.4f}")
    print(f"  Relevancy:     {avg_relevancy:.4f}")
    print(f"  Correctness:   {avg_correctness:.4f}")
    print(f"  Saved to: {results_path}")
    return summary


if __name__ == "__main__":
    embedder  = BGEEmbedder()
    store     = QdrantStore()
    dense     = DenseRetriever(embedder, store)
    bm25      = BM25Retriever()
    bm25.load()
    reranker  = BGEReranker()
    generator = OllamaGenerator(model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b"))
    pipeline  = RAGPipeline(dense, bm25, reranker, generator)

    run_evaluation(
        pipeline=pipeline,
        questions_path="data/golden/questions.json",
        results_path="data/golden/results_hybrid_reranked.json",
        run_name="hybrid_reranked",
    )