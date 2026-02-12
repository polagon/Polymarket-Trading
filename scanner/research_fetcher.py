"""
Astra V2 — Research Paper Fetcher
Recovers failed ResearchGate papers via Semantic Scholar API

Usage:
    python3.12 scanner/research_fetcher.py

Output:
    memory/recovered_papers.json — abstracts + PDF links for all recoverable papers
"""

import requests
import time
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# All ResearchGate papers that failed during research (from research_links_analysis.md)
FAILED_PAPERS = [
    {"id": "392756027", "title": "The Impact of High-Frequency Trading on Market Liquidity: A Mathematical Approach"},
    {"id": "356083585", "title": "FinRL-Podracer: High Performance and Scalable Deep Reinforcement Learning for Quantitative Finance"},
    {"id": "369612117", "title": "A Survey of Quantitative Trading Based on Artificial Intelligence"},
    {"id": "397926147", "title": "Automated Decision Making for Trading: A Comparative Analysis of Supervised and Reinforcement Learning"},
    {"id": "367019677", "title": "Quant 4.0: Engineering Quantitative Investment with Automated, Explainable, and Knowledge-driven Artificial Intelligence"},
    {"id": "348261598", "title": "Automated Creation of a High-Performing Algorithmic Trader via Deep Learning on Level-2 Limit Order Book Data"},
    {"id": "376601031", "title": "Quantitative Trading Wizardry: Crafting a Winning Robot"},
    {"id": "397426198", "title": "Automate Strategy Finding with LLM in Quant Investment"},
    {"id": "383917934", "title": "Automate Strategy Finding with LLM in Quant Investment"},  # Duplicate
    {"id": "387992428", "title": "AI-Driven Optimization of Financial Quantitative Trading Algorithms and Enhancement of Market Forecasting Capabilities"},
    {"id": "393237783", "title": "Deep Learning for Algorithmic Trading: A Systematic Review of Predictive Models and Optimization Strategies"},
    {"id": "366804382", "title": "Optimizing Automated Trading Systems with Deep Reinforcement Learning"},
    {"id": "395841128", "title": "Design and Implementation of a Multi-Strategy Algorithmic Trading Bot"},
    {"id": "348647950", "title": "mt5se: An Open Source Framework for Building Autonomous Trading Robots"},
    {"id": "396542021", "title": "AlphaQuanter: An End-to-End Tool-Orchestrated Agentic Reinforcement Learning Framework for Stock Trading"},
    {"id": "382282374", "title": "High-Frequency Quantitative Trading of Digital Currencies Based on Fusion of Deep Reinforcement Learning Models with Evolutionary Strategies"},
    {"id": "385709274", "title": "AI-Driven Optimization of Financial Quantitative Trading Algorithms"},
    {"id": "395587759", "title": "LSTM-Based Forex Trading Bot Using Python and MetaTrader 5: Design, Simulation and Evaluation"},
    {"id": "354800168", "title": "Towards Private On-Chain Algorithmic Trading"},
    {"id": "390737104", "title": "Algorithmic Trading Bots Using Artificial Intelligence"},
    {"id": "397370214", "title": "Application of Deep Reinforcement Learning in Quantitative Trading"},
    {"id": "395459365", "title": "Design and Evaluation of an AI-based Intelligent Trading Bot for the Foreign Exchange Market"},
    {"id": "383201516", "title": "Algorithmic Trading and Machine Learning: Advanced Techniques for Market Prediction and Strategy Development"},
    {"id": "390491406", "title": "Optimizing Automated Trading Systems Portfolios with Reinforcement Learning for Risk Control"},
    {"id": "395582308", "title": "The Influence of AI-Driven Bots and Algorithmic Trading on Bitcoin Price Volatility During COVID-19"},
    {"id": "387434546", "title": "The Automatic Cryptocurrency Trading System Using a Scalping Strategy"},
    {"id": "385748052", "title": "Reinforcement Learning Framework for Quantitative Trading"},
    {"id": "353770459", "title": "Algorithmic Trading Bot"},
    {"id": "375650502", "title": "Machine Learning-Based Quantitative Trading Strategies Across Different Time Intervals in the American Market"},
    {"id": "396748086", "title": "QuantEvolve: Automating Quantitative Strategy Discovery through Multi-Agent Evolutionary Framework"},
    {"id": "394622553", "title": "Systematic Review on Algorithmic Trading"},
    {"id": "387162838", "title": "AI-Powered Sentiment Analysis for Hedge Fund Trading Strategies"},
    {"id": "387168417", "title": "The Adoption of Quantitative AI Models in Hedge Fund Management"},
    {"id": "387170050", "title": "The Integration of AI in Hedge Fund Investment Strategies"},
    {"id": "377934753", "title": "Deep Reinforcement Learning Robots for Algorithmic Trading Considering Stock Market Conditions and US Interest Rates"},
    {"id": "388448293", "title": "Algorithmic Trading Bot Using Artificial Intelligence: Supertrend Strategy"},
    {"id": "387169141", "title": "AI in Hedge Fund Algorithmic Trading: Challenges and Opportunities"},
    {"id": "364025503", "title": "Automated Cryptocurrency Trading Bot Implementing Deep Reinforcement Learning"},
]


def fetch_via_semantic_scholar(title: str, retry: int = 3) -> dict | None:
    """
    Query Semantic Scholar API for paper by title.

    Returns:
        {
            "title": str,
            "abstract": str,
            "pdf_url": str | None,
            "authors": list,
            "year": int,
            "citations": int,
            "doi": str | None
        }
    """
    api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "fields": "title,abstract,openAccessPdf,authors,year,citationCount,externalIds",
        "limit": 1
    }

    for attempt in range(retry):
        try:
            response = requests.get(api_url, params=params, timeout=10)

            if response.status_code == 429:  # Rate limited
                logger.warning(f"⏳ Rate limited, waiting 10s...")
                time.sleep(10)
                continue

            if response.status_code == 200:
                data = response.json()
                if data.get("data") and len(data["data"]) > 0:
                    paper = data["data"][0]

                    # Extract PDF URL if available
                    pdf_url = None
                    if paper.get("openAccessPdf"):
                        pdf_url = paper["openAccessPdf"].get("url")

                    # Extract DOI
                    doi = None
                    if paper.get("externalIds"):
                        doi = paper["externalIds"].get("DOI")

                    return {
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract"),
                        "pdf_url": pdf_url,
                        "authors": [a.get("name") for a in paper.get("authors", [])],
                        "year": paper.get("year"),
                        "citations": paper.get("citationCount"),
                        "doi": doi,
                        "semantic_scholar_id": paper.get("paperId")
                    }

            return None

        except requests.RequestException as e:
            logger.error(f"❌ Request failed: {e}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue

    return None


def fetch_all_failed_papers() -> list[dict]:
    """
    Fetch all ResearchGate papers that failed via Semantic Scholar API.

    Returns list of results with status indicators.
    """
    results = []
    total = len(FAILED_PAPERS)

    logger.info(f"🔍 Fetching {total} failed ResearchGate papers via Semantic Scholar API...")
    logger.info(f"📊 Rate limit: 100 requests / 5 minutes (we'll stay well under)")
    logger.info("")

    found_count = 0
    pdf_count = 0

    for idx, paper in enumerate(FAILED_PAPERS, 1):
        logger.info(f"[{idx}/{total}] Searching: {paper['title'][:60]}...")

        result = fetch_via_semantic_scholar(paper["title"])

        if result:
            found_count += 1
            has_pdf = result.get("pdf_url") is not None
            if has_pdf:
                pdf_count += 1

            status = "✅ FOUND + PDF" if has_pdf else "📄 FOUND (abstract only)"
            logger.info(f"         {status}")

            results.append({
                "researchgate_id": paper["id"],
                "status": "FOUND_PDF" if has_pdf else "FOUND_ABSTRACT",
                **result
            })
        else:
            logger.info(f"         ❌ NOT FOUND")
            results.append({
                "researchgate_id": paper["id"],
                "status": "NOT_FOUND",
                "title": paper["title"],
                "abstract": None,
                "pdf_url": None
            })

        # Rate limiting: max 1 request per second (conservative)
        time.sleep(1.2)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"✅ Fetch complete!")
    logger.info(f"   Papers found:     {found_count}/{total} ({found_count/total*100:.0f}%)")
    logger.info(f"   PDFs available:   {pdf_count}/{total} ({pdf_count/total*100:.0f}%)")
    logger.info(f"   Abstracts only:   {found_count - pdf_count}")
    logger.info(f"   Not found:        {total - found_count}")
    logger.info("=" * 70)

    return results


def save_results(results: list[dict], output_path: str = "memory/recovered_papers.json"):
    """Save results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Results saved to: {output_file}")

    # Also create a human-readable summary
    summary_path = output_file.parent / "recovered_papers_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Recovered ResearchGate Papers Summary\n\n")
        f.write(f"**Total papers:** {len(results)}\n")
        f.write(f"**Found:** {sum(1 for r in results if r['status'].startswith('FOUND'))}\n")
        f.write(f"**With PDF:** {sum(1 for r in results if r['status'] == 'FOUND_PDF')}\n\n")

        f.write("---\n\n")

        # Group by status
        for status in ["FOUND_PDF", "FOUND_ABSTRACT", "NOT_FOUND"]:
            papers = [r for r in results if r["status"] == status]
            if papers:
                f.write(f"## {status.replace('_', ' ')} ({len(papers)} papers)\n\n")
                for paper in papers:
                    f.write(f"### {paper['title']}\n")
                    if paper.get("authors"):
                        f.write(f"**Authors:** {', '.join(paper['authors'][:3])}")
                        if len(paper['authors']) > 3:
                            f.write(f" et al.")
                        f.write("\n")
                    if paper.get("year"):
                        f.write(f"**Year:** {paper['year']}\n")
                    if paper.get("citations"):
                        f.write(f"**Citations:** {paper['citations']}\n")
                    if paper.get("pdf_url"):
                        f.write(f"**PDF:** {paper['pdf_url']}\n")
                    if paper.get("abstract"):
                        abstract_preview = paper['abstract'][:300] + "..." if len(paper.get('abstract', '')) > 300 else paper.get('abstract', '')
                        f.write(f"**Abstract:** {abstract_preview}\n")
                    f.write("\n---\n\n")

    logger.info(f"📝 Human-readable summary saved to: {summary_path}")


def analyze_results(results: list[dict]):
    """Print analysis of recovered papers."""
    logger.info("\n📊 ANALYSIS OF RECOVERED PAPERS:\n")

    # High-value papers (based on citations)
    found_papers = [r for r in results if r["status"].startswith("FOUND")]
    if found_papers:
        by_citations = sorted(found_papers, key=lambda x: x.get("citations", 0), reverse=True)
        logger.info("🏆 Top 5 Most-Cited Papers:")
        for i, paper in enumerate(by_citations[:5], 1):
            logger.info(f"   {i}. {paper['title'][:60]}...")
            logger.info(f"      Citations: {paper.get('citations', 0)} | Year: {paper.get('year', 'N/A')}")
            if paper.get("pdf_url"):
                logger.info(f"      PDF: {paper['pdf_url']}")
        logger.info("")

    # Papers with PDFs
    pdf_papers = [r for r in results if r["status"] == "FOUND_PDF"]
    if pdf_papers:
        logger.info(f"📄 {len(pdf_papers)} Papers with Direct PDF Access:")
        for paper in pdf_papers[:10]:  # Show first 10
            logger.info(f"   • {paper['title'][:50]}...")
        if len(pdf_papers) > 10:
            logger.info(f"   ... and {len(pdf_papers) - 10} more")
        logger.info("")

    # Key topics (based on titles)
    topics = {
        "Reinforcement Learning": sum(1 for r in results if "reinforcement learning" in r["title"].lower()),
        "LLM / AI": sum(1 for r in results if any(kw in r["title"].lower() for kw in ["llm", "ai-driven", "artificial intelligence"])),
        "Deep Learning": sum(1 for r in results if "deep learning" in r["title"].lower()),
        "Algorithmic Trading": sum(1 for r in results if "algorithmic trading" in r["title"].lower()),
        "Hedge Fund": sum(1 for r in results if "hedge fund" in r["title"].lower()),
        "Cryptocurrency": sum(1 for r in results if any(kw in r["title"].lower() for kw in ["crypto", "bitcoin"])),
    }

    logger.info("🔬 Topics Covered:")
    for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            logger.info(f"   • {topic}: {count} papers")
    logger.info("")


if __name__ == "__main__":
    logger.info("🚀 Astra V2 Research Paper Fetcher")
    logger.info("   Recovering failed ResearchGate papers via Semantic Scholar API")
    logger.info("")

    # Fetch all papers
    results = fetch_all_failed_papers()

    # Save results
    save_results(results)

    # Analyze what we found
    analyze_results(results)

    logger.info("✨ Done! Check memory/recovered_papers.json and memory/recovered_papers_summary.md")
