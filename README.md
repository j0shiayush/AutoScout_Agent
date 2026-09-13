# AutoScout Agent: Autonomous Lead Enrichment Pipeline

An asynchronous, Python-based AI agent designed to autonomously crawl company domains, handle dynamic JavaScript rendering, extract relevant subpages, and utilize LLMs to generate structured company intelligence.

## 🏗️ Architecture & Features

This pipeline was built to prioritize speed, token efficiency, and resilience against common scraping blockers.

- **Phase 1: Asynchronous Web Crawling (Playwright)**
  - Navigates dynamic JS-heavy websites using headless Chromium.
  - Automatically discovers and crawls high-value subpages (`/about`, `/team`, `/pricing`).
  - Implements grace-fallbacks: Switches from `networkidle` to `domcontentloaded` to prevent infinite hangs on sites with heavy background tracking (e.g., Postman, Vapi).
- **Phase 2: Context Pre-Processing (BeautifulSoup)**
  - Strips CSS, SVGs, scripts, and boilerplate navigation headers to extract clean text.
  - Significantly reduces LLM token consumption and eliminates raw HTML dumps.
- **Phase 3: Structured Extraction (Gemini 2.5 Flash + Pydantic)**
  - Forces strict JSON output schemas matching the target ICP data points.
  - **Bonus Feature:** Automatically tracks input/output tokens and estimates API cost per domain.

## 🛠️ Prerequisites

- Python 3.10+
- A Google Gemini API Key

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AutoScout_Agent.git
   cd AutoScout_Agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies and browser binaries:**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

## 🚀 Usage

Run the main pipeline to process the target domains (`postman.com`, `supabase.com`, `vapi.ai`):

```bash
python main.py
```

The script will output a structured intelligence report to `output/output.json` containing company overviews, ICPs, contact points, and key leadership.

## 📂 Project Structure

```text
AutoScout_Agent/
├── main.py               # Orchestrates the async scraping and LLM extraction phases
├── requirements.txt      # Project dependencies
├── README.md             # Setup and documentation
├── .env                  # Environment variables (git-ignored)
├── output/
│   └── output.json       # The final structured data deliverable
└── src/
    ├── __init__.py
    ├── scraper.py        # Playwright async browser logic and DOM cleaning
    └── extractor.py      # Gemini LLM generation, Pydantic schemas, and cost tracking
```