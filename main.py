import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from src.scraper import AutoScoutScraper
from src.extractor import AutoScoutExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

TARGET_DOMAINS = [
    "postman.com",
    "supabase.com",
    "vapi.ai",
]


async def process_domain(
    domain: str, scraper: AutoScoutScraper, extractor: AutoScoutExtractor
) -> dict:
    """Processes an individual domain through scraping and LLM extraction phases."""
    logging.info(f"\n{'='*50}\nSTARTING ENRICHMENT: {domain}\n{'='*50}")

    result = {
        "domain": domain,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "failed",
        "data": None,
    }

    try:
        scraped_text = await scraper.scrape_domain(domain)

        if not scraped_text:
            logging.warning(f"No usable content could be scraped from {domain}.")
            result["status"] = "no_content"
            return result

        intelligence = extractor.extract_intelligence(
            domain=domain, scraped_text=scraped_text
        )

        result["status"] = "success"
        result["data"] = intelligence
        logging.info(f"Successfully enriched domain: {domain}")

    except Exception as e:
        logging.error(f"Unexpected error while processing {domain}: {str(e)}")
        result["error"] = str(e)

    return result


async def main():
    if not os.getenv("GEMINI_API_KEY"):
        logging.error(
            "GEMINI_API_KEY is not set. Please create a .env file with your key."
        )
        sys.exit(1)

    logging.info("Initializing AutoScout Lead Enrichment Pipeline...")

    scraper = AutoScoutScraper(max_subpages=3, timeout=15000)
    extractor = AutoScoutExtractor()

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "output.json")

    results = []
    total_cost_usd = 0.0

    for domain in TARGET_DOMAINS:
        domain_result = await process_domain(domain, scraper, extractor)
        results.append(domain_result)

        if (
            domain_result.get("data")
            and "_metadata" in domain_result["data"]
            and "estimated_cost_usd" in domain_result["data"]["_metadata"]
        ):
            total_cost_usd += domain_result["data"]["_metadata"]["estimated_cost_usd"]

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n{'='*50}\nRUN COMPLETE")
    logging.info(f"Saved results for {len(results)} domains to: {output_file_path}")
    logging.info(f"Total Pipeline Estimated API Cost: ${total_cost_usd:.6f}")
    logging.info(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())