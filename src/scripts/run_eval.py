# scripts/run_eval.py
import sys
import os
import pandas as pd
import asyncio
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.infrastructure.embeddings.huggingface import HuggingFaceEmbedder
from src.infrastructure.vector_stores.qdrant_db import QdrantImpl
from src.application.services.rag_service import VectorStoreService, run_retrieval_service
from src.infrastructure.llm.local_llm_factory import LocalResourcesFactory

QDRANT_PATH = "qdrant_storage" 


def get_rag_service():
    # Re-instantiate your RAG service
    embedder = HuggingFaceEmbedder(model_name="all-MiniLM-L6-v2")
    db_impl = QdrantImpl(collection_name="rag_chunks", path=QDRANT_PATH) 
    return VectorStoreService(embedder, db_impl)

async def main():
    DATASET_PATH = "datasets/golden_dataset.csv"
    
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset not found. Run generate_dataset.py first.")
        return

    print("Load RAG Service...")
    rag_service = get_rag_service()
    
    print("Load Golden Dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # 1. Inference: Get answers from your RAG for every question
    questions = df['user_input'].tolist() # Ragas columns: user_input, reference
    ground_truths = df['reference'].tolist()
    
    answers = []
    contexts = []

    print(f"🧠 Running Inference on {len(questions)} questions...")
    for q in questions:
        # Run Retrieval
        results = run_retrieval_service(q, rag_service, top_k=3)
        # Simple concatenation of retrieved content for context
        retrieved_text = [c.content for c in results]
        # Let's assume the "Answer" is the top 1 chunk content for now.
        generated_answer = results[0].content if results else "No answer found"
        answers.append(generated_answer)
        contexts.append(retrieved_text)

    # 2. Prepare Data for Ragas Evaluation
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    eval_dataset = Dataset.from_dict(eval_data)
    # 3. Configure Local LLM for Evaluation Metrics
    # Note: Qwen 1.5B is very small for evaluation metrics. It might be unstable.
    llm = LocalResourcesFactory.get_generator_llm("qwen2.5:1.5b")
    embeddings = LocalResourcesFactory.get_embeddings()

    print("⚖️  Running Evaluation (Faithfulness & Relevancy)...")
    # We pass the local LLM to the metrics
    result = evaluate(
        eval_dataset,
        metrics=[answer_relevancy, faithfulness],
        llm=llm, 
        embeddings=embeddings
    )
    print("\n📊 Evaluation Results:")
    print(result)
    # Save results
    result_df = result.to_pandas()
    result_df.to_csv("datasets/evaluation_results.csv", index=False)
    print("💾 Results saved to datasets/evaluation_results.csv")

if __name__ == "__main__":
    asyncio.run(main())



 