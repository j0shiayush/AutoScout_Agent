import os
import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
        """
        Initializes the Gemini client. It automatically picks up GEMINI_API_KEY from the environment.
        """
        self.model_name = model_name
        self.client = genai.Client()
        
        self.cost_per_1m_input = 0.075 
        self.cost_per_1m_output = 0.30

    def _calculate_cost(self, usage_metadata) -> float:
        """
        Calculates the estimated API cost based on token usage.
        Bonus point requirement implemented here.
        """
        input_tokens = usage_metadata.prompt_token_count or 0
        output_tokens = usage_metadata.candidates_token_count or 0
        
        input_cost = (input_tokens / 1_000_000) * self.cost_per_1m_input
        output_cost = (output_tokens / 1_000_000) * self.cost_per_1m_output
        total_cost = input_cost + output_cost
        
        logging.info(f"Tokens Used - Input: {input_tokens} | Output: {output_tokens}")
        logging.info(f"Estimated Cost: ${total_cost:.6f}")
        return total_cost

    def extract_intelligence(self, domain: str, scraped_text: str) -> dict:
        """
        Passes the cleaned scraped text to the LLM and forces a structured Pydantic response.
        """
        if not scraped_text or len(scraped_text) < 50:
            logging.warning(f"Insufficient text scraped for {domain}. Returning default structure.")
            return self._get_empty_structure()

        prompt = f"""
        You are an expert AI data extraction agent. I have scraped the website context for {domain}.
        Your task is to analyze the provided text and extract the company intelligence perfectly 
        adhering to the requested JSON schema.
        
        If certain data points (like emails or leadership) are not found in the text, leave the arrays empty.
        Do not hallucinate or invent data. Base your answers entirely on the context below.
        
        --- SCRAPED CONTEXT ---
        {scraped_text}
        """

        try:
            logging.info(f"Sending {len(scraped_text)} chars to Gemini for extraction...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CompanyIntelligence,
                    temperature=0.1, # Low temperature for analytical extraction
                ),
            )
            
            extracted_data = json.loads(response.text)
            
            if response.usage_metadata:
                cost = self._calculate_cost(response.usage_metadata)
                extracted_data["_metadata"] = {"estimated_cost_usd": cost}
            
            return extracted_data
            
        except Exception as e:
            logging.error(f"LLM Extraction failed for {domain}: {e}")
            return self._get_empty_structure()
            
    def _get_empty_structure(self) -> dict:
        """Fallback mechanism if the LLM fails, ensuring the script never crashes."""
        return {
            "company_overview": "Data extraction failed or insufficient context.",
            "target_audience": "Unknown",
            "contact_points": [],
            "key_leadership": [],
            "confidence_score": 0.0
        }