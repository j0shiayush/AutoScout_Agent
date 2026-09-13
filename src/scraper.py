import asyncio
import logging
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import (
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RELEVANT_KEYWORDS = [
    "about",
    "team",
    "company",
    "leadership",
    "contact",
    "pricing",
    "people",
]

IGNORED_PATTERNS = [
    "blog",
    "docs",
    "documentation",
    "changelog",
    "careers",
    "jobs",
    "privacy",
    "terms",
    "legal",
    "status",
    "#",
    ".pdf",
    ".png",
    ".jpg",
]


class AutoScoutScraper:
    def __init__(self, max_subpages: int = 3, timeout: int = 15000):
        """
        :param max_subpages: Maximum number of relevant subpages to crawl per domain.
        :param timeout: Page load timeout in milliseconds.
        """
        self.max_subpages = max_subpages
        self.timeout = timeout

    def _clean_html_to_text(self, raw_html: str) -> str:
        """
        Strips CSS, scripts, SVGs, and boilerplate to extract
        clean text and optimize LLM token usage.
        """
        soup = BeautifulSoup(raw_html, "html.parser")

        for tag in soup(["script", "style", "svg", "noscript", "nav", "footer", "iframe"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())

    def _discover_relevant_subpages(self, base_url: str, raw_html: str) -> List[str]:
        """
        Parses anchor tags on the homepage, filters by domain and
        relevant keywords, and returns deduplicated subpage URLs.
        """
        soup = BeautifulSoup(raw_html, "html.parser")
        base_domain = urlparse(base_url).netloc.replace("www.", "")
        discovered_urls: Set[str] = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            link_text = a_tag.get_text(strip=True).lower()
            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)

            target_domain = parsed_url.netloc.replace("www.", "")
            if target_domain != base_domain:
                continue

            clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}".rstrip("/")
            path_lower = parsed_url.path.lower()

            if clean_url == base_url.rstrip("/"):
                continue

            if any(ignored in path_lower for ignored in IGNORED_PATTERNS):
                continue

            is_relevant = any(
                keyword in path_lower or keyword in link_text for keyword in RELEVANT_KEYWORDS
            )

            if is_relevant:
                discovered_urls.add(clean_url)
                if len(discovered_urls) >= self.max_subpages:
                    break

        return list(discovered_urls)

    async def _fetch_page_text(self, page: Page, url: str) -> str:
        """
        Loads a single page with error handling and returns its cleaned text.
        """
        try:
            logging.info(f"Navigating to: {url}")
            await page.goto(url, wait_until="networkidle", timeout=self.timeout)
            content = await page.content()
            return self._clean_html_to_text(content)
        except PlaywrightTimeoutError:
            logging.warning(f"Timeout on {url}. Falling back to DOMContentLoaded.")
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=5000)
                content = await page.content()
                return self._clean_html_to_text(content)
            except Exception as e:
                logging.warning(f"Fallback failed for {url}: {e}")
                return ""
        except Exception as e:
            logging.warning(f"Failed to fetch {url}: {e}")
            return ""

    async def scrape_domain(self, domain: str) -> str:
        """
        Entry point: Crawls the homepage, discovers relevant subpages (/about, /team, etc.),
        extracts clean text, and bundles everything into structured context.
        """
        base_url = f"https://{domain}" if not domain.startswith("http") else domain
        logging.info(f"\n--- Starting crawl for target: {base_url} ---")

        combined_context: List[str] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
            )

            try:
                logging.info(f"Fetching homepage: {base_url}")
                try:
                    await context.goto(base_url, wait_until="networkidle", timeout=self.timeout)
                except PlaywrightTimeoutError:
                    logging.warning(f"networkidle timeout on {base_url}. Falling back to domcontentloaded.")
                    await context.goto(base_url, wait_until="domcontentloaded", timeout=10000)
                    await context.wait_for_timeout(2000) 

                homepage_html = await context.content()
                homepage_text = self._clean_html_to_text(homepage_html)

                combined_context.append(f"=== SOURCE: Homepage ({base_url}) ===\n{homepage_text}")

                subpages = self._discover_relevant_subpages(base_url, homepage_html)
                logging.info(f"Discovered {len(subpages)} relevant subpages: {subpages}")

                for sub_url in subpages:
                    sub_text = await self._fetch_page_text(context, sub_url)
                    if sub_text:
                        combined_context.append(f"\n=== SOURCE: Subpage ({sub_url}) ===\n{sub_text}")

            except PlaywrightTimeoutError:
                logging.error(f"Homepage load timed out for {base_url}. Handled gracefully.")
            except Exception as e:
                logging.error(f"Error while scraping {base_url}: {e}")
            finally:
                await browser.close()

        final_context = "\n\n".join(combined_context)
        logging.info(f"Finished crawling {domain}. Total text length: {len(final_context)} chars.")
        return final_context