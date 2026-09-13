import os
import json
import logging
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from ddgs import DDGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 1. Define the Strict Output Schema using Pydantic
class LeadershipMember(BaseModel):
    name: str = Field(description="Full name of the team member or founder.")
    role: str = Field(description="Job title or role (e.g., CEO, Founder).")
    linkedin_url: Optional[str] = Field(description="LinkedIn URL if discoverable in the text, otherwise null.")

class CompanyIntelligence(BaseModel):
    company_overview: str = Field(description="A concise 2-sentence summary of what the company does based on the text.")
    target_audience: str = Field(description="Who their product is built for (e.g., 'Developers building backend applications').")
    contact_points: List[str] = Field(description="List of generic or public emails found (e.g., contact@, sales@).")
    key_leadership: List[LeadershipMember] = Field(description="List of key leadership or team members.")
    confidence_score: float = Field(description="An estimated score between 0.0 and 1.0 indicating the quality/completeness of the extracted data.")

class AutoScoutExtractor:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client()
        
        # Approximate Cost for Gemini Flash models (per 1 Million tokens)
        self.cost_per_1m_input = 0.075 
        self.cost_per_1m_output = 0.30

    def _calculate_cost(self, usage_metadata) -> float:
        """Calculates the estimated API cost based on token usage."""
        input_tokens = usage_metadata.prompt_token_count or 0
        output_tokens = usage_metadata.candidates_token_count or 0
        
        input_cost = (input_tokens / 1_000_000) * self.cost_per_1m_input
        output_cost = (output_tokens / 1_000_000) * self.cost_per_1m_output
        total_cost = input_cost + output_cost
        
        logging.info(f"Tokens Used - Input: {input_tokens} | Output: {output_tokens}")
        return total_cost

    def _agentic_linkedin_search(self, name: str, domain: str) -> Optional[str]:
        """
        Custom tool-calling function: Uses a search engine to find external 
        LinkedIn URLs for founders if not found on the direct website.
        """
        query = f"{name} {domain} site:linkedin.com/in/"
        logging.info(f"Agentic Loop Triggered: Searching web for '{name}' LinkedIn...")
        
        try:
            # Adding a brief sleep to respect search engine rate limits
            time.sleep(1.5)
            results = DDGS().text(query, max_results=1)
            if results and len(results) > 0:
                discovered_url = results[0].get("href")
                logging.info(f"Found missing LinkedIn URL: {discovered_url}")
                return discovered_url
        except Exception as e:
            logging.warning(f"Search tool failed for {name}: {e}")
            
        return None

    def extract_intelligence(self, domain: str, scraped_text: str) -> dict:
        """
        Executes a multi-step agentic extraction:
        1. Pydantic DOM extraction.
        2. Validation loop for missing LinkedIn profiles.
        3. Web search tool calling to enrich missing data.
        """
        if not scraped_text or len(scraped_text) < 50:
            logging.warning(f"Insufficient text scraped for {domain}.")
            return self._get_empty_structure()

        prompt = f"""
        You are an expert AI data extraction agent. I have scraped the website context for {domain}.
        Extract the company intelligence perfectly adhering to the requested JSON schema.
        If a LinkedIn URL is missing, set it to null.
        
        --- SCRAPED CONTEXT ---
        {scraped_text}
        """

        try:
            # Node 1: Base Extraction
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CompanyIntelligence,
                    temperature=0.1, 
                ),
            )
            
            extracted_data = json.loads(response.text)
            
            # Node 2: Agentic Search Enrichment Loop
            # If the LLM missed a LinkedIn URL, the agent autonomously searches for it
            for leader in extracted_data.get("key_leadership", []):
                if leader.get("name") and not leader.get("linkedin_url"):
                    found_url = self._agentic_linkedin_search(leader["name"], domain)
                    if found_url:
                        leader["linkedin_url"] = found_url
                        # Boost confidence score slightly since we successfully enriched the data
                        extracted_data["confidence_score"] = min(1.0, extracted_data.get("confidence_score", 0.0) + 0.05)
            
            # Record Cost
            if response.usage_metadata:
                cost = self._calculate_cost(response.usage_metadata)
                extracted_data["_metadata"] = {"estimated_cost_usd": cost}
            
            return extracted_data
            
        except Exception as e:
            logging.error(f"LLM Extraction failed for {domain}: {e}")
            return self._get_empty_structure()
            
    def _get_empty_structure(self) -> dict:
        return {
            "company_overview": "Data extraction failed.",
            "target_audience": "Unknown",
            "contact_points": [],
            "key_leadership": [],
            "confidence_score": 0.0
        }