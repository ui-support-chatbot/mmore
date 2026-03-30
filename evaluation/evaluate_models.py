import os
import sys
import time
import json
import logging
import requests
import pandas as pd
from datasets import Dataset

# Ensure MMORE src is accessible
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from mmore.rag.pipeline import RAGPipeline, RAGConfig
from mmore.utils import load_config

# ----------------- Configuration -----------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "sk_rektor_rag_docker.yaml")
# Now using the 10-question verified golden dataset sample you liked
DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset_sample.json")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "evaluation_results_full_local.csv")

# List of models to evaluate against
MODELS_TO_TEST = [
    "qwen3.5:9b",
    "gemma3:4b",
    "gemma3:1B",
    "ministral-3:3b",
    "ministral-3:8b",
    "llama3.2:3b"
]

# The local Ollama model that will ACT AS THE JUDGE to score the generating models.
# Reverting to Qwen 2.5 7B which has proven consistent for Ragas extractions.
JUDGE_MODEL = "qwen2.5:7b" 

COLLECTION_NAME = "sk_rektor_docs"

# -------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pull_ollama_model(model_name: str):
    """Pulls an Ollama model over API if it's not fully downloaded yet."""
    logger.info(f"Checking/Pulling model: {model_name}...")
    url = "http://localhost:11434/api/pull"
    payload = {"name": model_name}
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "status" in data:
                    print(f"[{model_name}] {data['status']}", end="\r")
        print()
    except Exception as e:
        logger.error(f"Failed to pull {model_name}: {e}")

def main():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Golden dataset not found at {DATASET_PATH}. Please ensure it exists.")
        sys.exit(1)

    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config path not found at {CONFIG_PATH}.")
        sys.exit(1)

    # 1. Load the golden questions & answers
    logger.info(f"Loading golden dataset from {DATASET_PATH}")
    with open(DATASET_PATH, "r") as f:
        golden_data = json.load(f)

    # 2. Setup RAGAS Evaluator Components
    logger.info(f"Initializing Local Judge LLM ({JUDGE_MODEL}) and Judge Embeddings (BGE-M3)...")
    
    # Pre-pull the judge model
    pull_ollama_model(JUDGE_MODEL)
    
    # Initialize the fully local evaluator LLM with strict 0.0 temperature
    # to prevent "overthinking" and force it to act strictly as a judge.
    judge_llm = ChatOllama(model=JUDGE_MODEL, base_url="http://localhost:11434", temperature=0.0)
    
    # Ragas needs an embedding model to evaluate metrics
    judge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # WE ARE MEASURING EVERYTHING NOW
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall()
    ]

    all_results = []

    # 3. Load Base Configuration
    import yaml
    from dacite import from_dict
    with open(CONFIG_PATH, "r") as f:
        raw_config = yaml.safe_load(f)
    rag_config = from_dict(data_class=RAGConfig, data=raw_config["rag"])

    for model_name in MODELS_TO_TEST:
        logger.info(f"=== Starting Evaluation for {model_name} ===")
        pull_ollama_model(model_name)

        # Update RAGConfig context to use the sequence model
        rag_config.llm.llm_name = model_name
        
        try:
            pipeline = RAGPipeline.from_config(rag_config)
        except Exception as e:
            logger.error(f"Failed to initialize pipeline for {model_name}: {e}")
            continue

        model_results = {
            "user_input": [],
            "retrieved_contexts": [],
            "response": [],
            "reference": [],
            "generation_time": []
        }

        logger.info("Generating answers for the dataset...")
        for i, item in enumerate(golden_data):
            question = item["question"]
            # The key in golden_dataset_sample.json is "answer"
            reference = item.get("answer", "")

            query_dict = {
                "input": question,
                "collection_name": COLLECTION_NAME,
            }

            start_time = time.time()
            try:
                # pipeline with return_dict=True will return the answer + context from the LCEL chain
                res = pipeline(queries=query_dict, return_dict=True)[0]
                
                ans = res.get("answer", "")
                ctx = res.get("context", "")
                
            except Exception as e:
                logger.error(f"Error during generation for question {i+1} ('{question}'): {e}")
                continue
                
            end_time = time.time()
            latency = end_time - start_time
            logger.info(f"  -> [{model_name}] Answered Q{i+1}/{len(golden_data)} in {latency:.2f}s")

            model_results["user_input"].append(question)
            model_results["response"].append(ans)
            model_results["retrieved_contexts"].append([ctx] if ctx else [])
            # Map the golden truth
            model_results["reference"].append(reference)
            model_results["generation_time"].append(latency)

        if not model_results["user_input"]:
            logger.warning(f"No results generated for {model_name}. Skipping evaluation.")
            continue

        # Convert dictionary to HuggingFace Dataset required by RAGAS
        dataset = Dataset.from_dict({
            "user_input": model_results["user_input"],
            "retrieved_contexts": model_results["retrieved_contexts"],
            "response": model_results["response"],
            "reference": model_results["reference"]
        })

        eval_dataset = EvaluationDataset.from_hf_dataset(dataset)

        logger.info(f"Running RAGAS Metrics for {model_name}...")
        try:
            evaluation_result = evaluate(
                dataset=eval_dataset,
                metrics=metrics,
                llm=judge_llm,
                embeddings=judge_embeddings
            )

            # Extract numeric averages for final row
            df_res = evaluation_result.to_pandas()
            
            # SAVE THE DETAILED ANSWERS & INDIVIDUAL QUESTION SCORES
            safe_model_name = model_name.replace(":", "_").replace(".", "_")
            detailed_csv_path = os.path.join(os.path.dirname(__file__), f"details_{safe_model_name}.csv")
            df_res.to_csv(detailed_csv_path, index=False)
            logger.info(f"Saved ALL raw answers and specific metrics for {model_name} to {detailed_csv_path}")

            numeric_cols = df_res.select_dtypes(include='number').columns
            avg_scores = df_res[numeric_cols].mean().to_dict()

            avg_time = sum(model_results["generation_time"]) / len(model_results["generation_time"])

            row_result = {
                "Model": model_name,
                "Latency (s)": round(avg_time, 2),
                **{k: round(v, 4) for k, v in avg_scores.items()}
            }
            all_results.append(row_result)
            logger.info(f"Scores for {model_name}: {row_result}")
            
            # Incrementally save the summary CSV after EVERY model finishes safely
            df_final = pd.DataFrame(all_results)
            df_final.to_csv(OUTPUT_CSV, index=False)
            logger.info(f"Incrementally updated {OUTPUT_CSV} safely.")
        
        except Exception as e:
            logger.error(f"RAGAS evaluation failed for {model_name}: {e}")

    # 4. Final summary print
    if all_results:
        df_final = pd.DataFrame(all_results)
        logger.info("Evaluation complete! All models are fully saved.")
        print("\n=== FINAL RESULTS ===")
        print(df_final.to_string(index=False))
    else:
        logger.warning("No successful evaluations to save.")

if __name__ == "__main__":
    main()
