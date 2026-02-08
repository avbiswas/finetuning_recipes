import re
import time

import arxiv
import requests
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader


def query_arxiv(query: str = "transformer large language model", max_results: int = 50) -> list[dict]:
    """Query arxiv and return 2025 papers, sorted by relevance."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in search.results():
        # if result.published.year < 2025:
        #     continue
        # strip version suffix (e.g. 2501.12345v2 -> 2501.12345)
        # result.entry_id is the full URL, e.g. http://arxiv.org/abs/2101.12345v1
        arxiv_id = re.sub(r"v\d+$", "", result.entry_id.split("/")[-1])
        papers.append({
            "arxiv_id": arxiv_id,
            "title": " ".join(result.title.split()),
            "published": result.published,
        })

    return papers


def download_paper(arxiv_id: str) -> str:
    """Download a PDF from arxiv and return extracted text via pypdf."""
    url = f"https://export.arxiv.org/pdf/{arxiv_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    reader = PdfReader(BytesIO(resp.content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def main():
    output_dir = Path("papers")
    output_dir.mkdir(exist_ok=True)

    # date-scoped query for 2025 transformer / LLM papers
    # Fix: Use explicit date range instead of wildcard '*' which causes 500 errors
    query = '(transformer OR LLM OR "large language model" OR "attention" OR "mixture of experts") AND submittedDate:[20250101 TO 20260101]'
    print(f"Querying arxiv with: {query}\n")
    papers = query_arxiv(query=query, max_results=50)

    # fallback to a broader query if the date filter is too restrictive
    if len(papers) < 10:
        print(f"Date-filtered query returned only {len(papers)} results — broadening search...\n")
        papers = query_arxiv(query="transformer large language model", max_results=200)
    
    selected = papers
    print(f"Downloading {len(selected)} papers\n")
    print(selected)
    for i, paper in enumerate(selected, 1):
        arxiv_id = paper["arxiv_id"]
        filepath = output_dir / f"{arxiv_id}.txt"

        print(f"[{i}/{len(selected)}] {arxiv_id}  {paper['title'][:72]}")

        if filepath.exists():
            print(f"         -> exists, skipping\n")
            continue

        try:
            text = download_paper(arxiv_id)
            filepath.write_text(text, encoding="utf-8")
            print(f"         -> {filepath}  ({len(text):,} chars)\n")
        except Exception as e:
            print(f"         -> failed: {e}\n")

        time.sleep(1)  # polite delay between requests


if __name__ == "__main__":
    main()
