import json
import logging
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import streamlit as st

from config import config
from llm import get_chat_llm
from rag_chain import DocumentRAGChain
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class ContextRelevanceScore(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0 indicating if context contains the answer")
    reasoning: str = Field(description="Reasoning for the score")

class FaithfulnessScore(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0 indicating if the answer is factually supported by the context")
    reasoning: str = Field(description="Reasoning for the score")

class AnswerRelevanceScore(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0 indicating if the answer addresses the question")
    reasoning: str = Field(description="Reasoning for the score")

class RAGEvaluator:
    def __init__(self):
        self.llm = get_chat_llm()
        self.context_parser = JsonOutputParser(pydantic_object=ContextRelevanceScore)
        self.faithfulness_parser = JsonOutputParser(pydantic_object=FaithfulnessScore)
        self.answer_parser = JsonOutputParser(pydantic_object=AnswerRelevanceScore)
        
        self.context_prompt = ChatPromptTemplate.from_template(
            "Evaluate if the provided context contains sufficient information to answer the question.\n"
            "Question: {question}\n"
            "Context: {context}\n"
            "Output JSON with 'score' (0.0 to 1.0) and 'reasoning'.\n{format_instructions}"
        )
        self.faithfulness_prompt = ChatPromptTemplate.from_template(
            "Evaluate if the generated answer is faithful to the provided context. It should not contain made up facts outside the context.\n"
            "Context: {context}\n"
            "Answer: {answer}\n"
            "Output JSON with 'score' (0.0 to 1.0) and 'reasoning'.\n{format_instructions}"
        )
        self.answer_prompt = ChatPromptTemplate.from_template(
            "Evaluate if the generated answer addresses the user's question accurately based on the ground truth.\n"
            "Question: {question}\n"
            "Answer: {answer}\n"
            "Ground Truth: {ground_truth}\n"
            "Output JSON with 'score' (0.0 to 1.0) and 'reasoning'.\n{format_instructions}"
        )

    def evaluate_turn(self, question: str, answer: str, context: str, ground_truth: str) -> Dict[str, Any]:
        results = {}
        
        # Context Relevance
        try:
            chain = self.context_prompt | self.llm | self.context_parser
            res = chain.invoke({"question": question, "context": context, "format_instructions": self.context_parser.get_format_instructions()})
            results["context_relevance"] = res["score"]
            results["context_relevance_reasoning"] = res["reasoning"]
        except Exception as e:
            logger.error(f"Context relevance eval failed: {e}")
            results["context_relevance"] = 0.0
            
        # Faithfulness
        try:
            chain = self.faithfulness_prompt | self.llm | self.faithfulness_parser
            res = chain.invoke({"context": context, "answer": answer, "format_instructions": self.faithfulness_parser.get_format_instructions()})
            results["faithfulness"] = res["score"]
            results["faithfulness_reasoning"] = res["reasoning"]
        except Exception as e:
            logger.error(f"Faithfulness eval failed: {e}")
            results["faithfulness"] = 0.0
            
        # Answer Relevance
        try:
            chain = self.answer_prompt | self.llm | self.answer_parser
            res = chain.invoke({"question": question, "answer": answer, "ground_truth": ground_truth, "format_instructions": self.answer_parser.get_format_instructions()})
            results["answer_relevance"] = res["score"]
            results["answer_relevance_reasoning"] = res["reasoning"]
        except Exception as e:
            logger.error(f"Answer relevance eval failed: {e}")
            results["answer_relevance"] = 0.0
            
        return results

    def run_evaluation_suite(self, dataset_path: str, rag_chain: DocumentRAGChain, num_samples: int = 5, progress_callback=None) -> List[Dict[str, Any]]:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        dataset = dataset[:num_samples]
        results = []
        
        for i, item in enumerate(dataset):
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            # Run RAG
            rag_res = rag_chain.query(question=question, k=config.TOP_K_RESULTS)
            answer = rag_res.get("answer", "")
            
            # Combine sources into a single context string
            context_parts = []
            for src in rag_res.get("sources", []):
                context_parts.append(src.get("preview", ""))
            context = "\n".join(context_parts)
            
            # Evaluate
            eval_res = self.evaluate_turn(question, answer, context, ground_truth)
            
            # Combine
            row = {
                "question": question,
                "answer": answer,
                "context_relevance": eval_res.get("context_relevance", 0.0),
                "faithfulness": eval_res.get("faithfulness", 0.0),
                "answer_relevance": eval_res.get("answer_relevance", 0.0)
            }
            results.append(row)
            
            if progress_callback:
                progress_callback(i + 1, len(dataset))
                
        return results
