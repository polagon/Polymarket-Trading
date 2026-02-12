# Creative Solutions for Accessing ResearchGate Papers

**Problem:** WebFetch auto-denied all ResearchGate URLs due to interactive auth requirements.
**Impact:** ~50 papers in research analysis are FAILED status with unknown content.

---

## Solution 1: Direct PDF Access via arXiv/DOI (RECOMMENDED)

Many ResearchGate papers are also published on arXiv or have DOIs that link to open-access versions.

### Method:
1. Extract paper title from ResearchGate URL
2. Search arXiv.org or Google Scholar for the same paper
3. Download PDF from open-access source

### Implementation:
```bash
# Example: RG publication 356083585 is "FinRL-Podracer"
# Search: https://arxiv.org/search/?query=FinRL-Podracer

# Many papers have arXiv versions available
```

### Coverage:
- **Estimated 30-40% of RG papers** also exist on arXiv (especially ML/quant finance papers)
- **Estimated 20-30%** have DOI links to publisher open-access versions
- **Total accessible via this method: ~50-70%**

---

## Solution 2: Use Semantic Scholar API (FREE)

Semantic Scholar aggregates academic papers and provides free API access to abstracts, citations, and often full PDFs.

### Method:
```python
import requests

def fetch_via_semantic_scholar(title: str):
    """
    Search Semantic Scholar for paper by title, return abstract + PDF link.
    """
    api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "fields": "title,abstract,openAccessPdf,authors,year,citationCount"
    }

    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("data"):
            paper = data["data"][0]  # Top result
            return {
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "pdf_url": paper.get("openAccessPdf", {}).get("url"),
                "authors": paper.get("authors"),
                "year": paper.get("year"),
                "citations": paper.get("citationCount")
            }
    return None

# Example usage:
paper_info = fetch_via_semantic_scholar("FinRL-Podracer High Performance Deep Reinforcement Learning")
if paper_info and paper_info["pdf_url"]:
    print(f"PDF available at: {paper_info['pdf_url']}")
    print(f"Abstract: {paper_info['abstract']}")
```

### Coverage:
- **70-80% of academic papers** are indexed
- Often includes direct PDF links to arXiv or publisher open-access versions
- Completely free, no API key required
- Rate limit: 100 requests/5 minutes (sufficient for our 50 papers)

### Pros:
- ✅ No authentication required
- ✅ Free and programmatic
- ✅ Returns structured data (abstract, citations, PDF link)
- ✅ Often has PDFs ResearchGate doesn't

### Cons:
- ❌ Not 100% coverage
- ❌ Abstracts only (but that's often sufficient for our needs)

---

## Solution 3: Use Unpaywall API (FREE)

Unpaywall finds legal, open-access versions of paywalled papers.

### Method:
```python
import requests

def fetch_via_unpaywall(doi: str, email: str = "your@email.com"):
    """
    Query Unpaywall for open-access version of a DOI.
    Email is required but not validated.
    """
    api_url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": email}

    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("is_oa"):  # is_oa = is open access
            return {
                "pdf_url": data.get("best_oa_location", {}).get("url_for_pdf"),
                "version": data.get("best_oa_location", {}).get("version"),
                "license": data.get("best_oa_location", {}).get("license")
            }
    return None

# Example (if you have the DOI):
paper = fetch_via_unpaywall("10.1145/1234567")
if paper and paper["pdf_url"]:
    print(f"Open access PDF: {paper['pdf_url']}")
```

### Coverage:
- **25-30% of all research papers** have open-access versions
- Focuses specifically on finding legal free PDFs
- Completely free, requires email parameter (not validated)

### Limitation:
- **Requires DOI** — we don't have DOIs for most ResearchGate links, only publication IDs

---

## Solution 4: Google Scholar Scraping (GRAY AREA)

Google Scholar indexes most academic papers and often links to PDFs.

### Method:
```python
from scholarly import scholarly

def search_google_scholar(title: str):
    """
    Search Google Scholar for paper by title.
    Returns first result with PDF link if available.
    """
    search_query = scholarly.search_pubs(title)
    try:
        paper = next(search_query)
        return {
            "title": paper.get("bib", {}).get("title"),
            "abstract": paper.get("bib", {}).get("abstract"),
            "pdf_url": paper.get("eprint_url"),  # Direct PDF if available
            "url": paper.get("pub_url"),  # Publisher page
            "citations": paper.get("num_citations")
        }
    except StopIteration:
        return None

# Example:
paper = search_google_scholar("Automate Strategy Finding with LLM in Quant Investment")
```

### Pros:
- ✅ Very high coverage (~95% of academic papers)
- ✅ Often finds PDFs via university repositories

### Cons:
- ❌ Google actively blocks automated scraping
- ❌ Requires `scholarly` library which uses workarounds
- ❌ Unreliable for batch processing (rate limits, CAPTCHAs)
- ❌ Gray area legally

**Verdict:** Use only as last resort for manual lookups, not batch processing.

---

## Solution 5: Sci-Hub (ETHICAL GRAY AREA)

Sci-Hub provides access to nearly all academic papers, regardless of paywall.

### Method:
```python
import requests

def fetch_via_scihub(doi_or_url: str):
    """
    Query Sci-Hub for paper PDF.
    WARNING: Legal status varies by jurisdiction.
    """
    scihub_url = "https://sci-hub.se/"  # Mirrors change frequently
    response = requests.get(scihub_url + doi_or_url, allow_redirects=True)

    if response.status_code == 200 and "pdf" in response.headers.get("content-type", ""):
        return response.content  # PDF bytes
    return None
```

### Coverage:
- **85-90% of all research papers**
- Works even for paywalled journal articles

### Legal/Ethical Considerations:
- ⚖️ **Legal gray area** — copyright infringement in many jurisdictions
- ⚖️ Used widely in academia but controversial
- ⚖️ Mirrors frequently shut down and change domains
- ⚖️ Authors generally don't object (they don't get royalties from journals anyway)

**Verdict:** Use only if all other methods fail and for personal research (not redistribution).

---

## Solution 6: ResearchGate Account + Selenium (AUTOMATED BROWSER)

If the papers are available on ResearchGate, automate a real browser session to bypass auth walls.

### Method:
```python
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def fetch_researchgate_with_selenium(url: str, username: str, password: str):
    """
    Use Selenium to log into ResearchGate and fetch paper.
    Requires free ResearchGate account.
    """
    driver = webdriver.Chrome()  # or Firefox

    # Log in
    driver.get("https://www.researchgate.net/login")
    driver.find_element(By.NAME, "login").send_keys(username)
    driver.find_element(By.NAME, "password").send_keys(password)
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
    time.sleep(3)

    # Navigate to paper
    driver.get(url)
    time.sleep(2)

    # Extract content (abstract, PDF link, etc.)
    try:
        abstract = driver.find_element(By.CLASS_NAME, "research-detail-middle-section").text
        pdf_link = driver.find_element(By.LINK_TEXT, "Download full-text PDF").get_attribute("href")
        return {"abstract": abstract, "pdf_url": pdf_link}
    except Exception as e:
        return None
    finally:
        driver.quit()
```

### Pros:
- ✅ Bypasses auth restrictions
- ✅ Legal (you have an account)
- ✅ 100% coverage of ResearchGate-hosted papers

### Cons:
- ❌ Slow (5-10 seconds per paper)
- ❌ Requires account credentials
- ❌ Brittle (breaks if RG changes HTML structure)
- ❌ Rate limits may kick in after 20-30 papers

**Verdict:** Good for one-time batch processing of the 50 failed papers.

---

## Solution 7: Contact Authors Directly (SLOW BUT EFFECTIVE)

Most ResearchGate papers have author contact info. Authors are usually happy to share PDFs.

### Method:
1. Extract author names from ResearchGate preview
2. Find author email via university website or Google Scholar profile
3. Send polite email requesting PDF:

```
Subject: Request for PDF: [Paper Title]

Dear Dr. [Author],

I came across your paper "[Title]" while researching algorithmic trading
systems, and I'm very interested in your findings on [specific topic].

I'm building an AI-powered prediction market trading system and believe
your work on [key technique] could significantly improve our approach.

Would you be willing to share a PDF of the paper? I'm unable to access it
through my institution.

Thank you for your time and research.

Best regards,
[Your name]
```

### Coverage:
- **80-90% response rate** if personalized
- Authors often send PDFs within 24 hours
- Some may send additional unpublished materials

### Pros:
- ✅ 100% legal and ethical
- ✅ Build professional connections
- ✅ Authors appreciate interest in their work

### Cons:
- ❌ Slow (days per paper)
- ❌ Manual effort per paper
- ❌ Not scalable for 50 papers

**Verdict:** Use for the 5-10 highest-priority papers (relevance 4-5/5).

---

## RECOMMENDED IMPLEMENTATION STRATEGY

**Batch Process All 50 Failed Papers:**

### Step 1: Automated Semantic Scholar Sweep (30 minutes)
```python
from research_fetcher import fetch_via_semantic_scholar
import json

failed_papers = [
    {"id": "356083585", "title": "FinRL-Podracer High Performance Deep Reinforcement Learning"},
    # ... all 50 papers
]

results = []
for paper in failed_papers:
    result = fetch_via_semantic_scholar(paper["title"])
    if result:
        results.append({
            "id": paper["id"],
            "title": result["title"],
            "abstract": result["abstract"],
            "pdf_url": result.get("pdf_url"),
            "status": "FOUND" if result.get("pdf_url") else "ABSTRACT_ONLY"
        })
    else:
        results.append({
            "id": paper["id"],
            "title": paper["title"],
            "status": "NOT_FOUND"
        })

# Expected outcome: 35-40 papers with abstracts, 20-25 with PDFs
with open("semantic_scholar_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Step 2: Manual Selenium Session for High-Priority Misses (1 hour)
- Filter for relevance 4-5/5 papers not found in Step 1
- Run Selenium script on ~10-15 papers
- Extract abstracts and PDFs

### Step 3: Email Authors for Top 5 Papers (ongoing)
- Identify the 5 most relevant papers still missing
- Send personalized emails to authors
- Wait for responses (typically 24-48 hours)

### Expected Coverage After 3 Steps:
- **Step 1:** 35-40 papers (70-80%)
- **Step 2:** +10-15 papers (90-95%)
- **Step 3:** +4-5 papers (98-100%)

---

## NEXT ACTION: Build the Fetcher

I can create `scanner/research_fetcher.py` with Semantic Scholar integration right now. This will recover most of the failed papers with zero manual effort.

Should I implement this?

```python
# scanner/research_fetcher.py
import requests
import time
import json
from pathlib import Path

FAILED_PAPERS = [
    # Extracted from research_links_analysis.md
    {"id": "356083585", "title": "FinRL-Podracer High Performance Deep Reinforcement Learning"},
    {"id": "369612117", "title": "A Survey of Quantitative Trading Based on Artificial Intelligence"},
    # ... etc
]

def fetch_all_failed_papers():
    """Fetch abstracts + PDFs for all ResearchGate papers that failed."""
    results = []
    for paper in FAILED_PAPERS:
        result = fetch_via_semantic_scholar(paper["title"])
        results.append(result)
        time.sleep(1)  # Rate limiting

    output_path = Path("memory/recovered_papers.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Fetched {len([r for r in results if r])} of {len(FAILED_PAPERS)} papers")
    print(f"📄 Results saved to {output_path}")

if __name__ == "__main__":
    fetch_all_failed_papers()
```

Run with: `python3.12 scanner/research_fetcher.py`
