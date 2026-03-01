import re
import time
import concurrent.futures

import arxiv
import requests
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader


output_dir = Path("papers")
output_dir.mkdir(exist_ok=True)


def query_arxiv(
    query: str = "transformer large language model", max_results: int = 50
) -> list[dict]:
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
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": " ".join(result.title.split()),
                "published": result.published,
            }
        )

    return papers


def download_paper(arxiv_id: str) -> str:
    """Download a PDF from arxiv and return extracted text via pypdf."""
    url = f"https://export.arxiv.org/pdf/{arxiv_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    reader = PdfReader(BytesIO(resp.content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def process_paper(arxiv_id: str, i: int, total: int, title: str = None):
    filepath = output_dir / f"{arxiv_id}.txt"
    title_str = f"  {title[:72]}" if title else ""
    print(f"[{i}/{total}] {arxiv_id}{title_str}")

    if filepath.exists():
        print(f"         -> {arxiv_id} exists, skipping\n")
        return

    try:
        text = download_paper(arxiv_id)
        filepath.write_text(text, encoding="utf-8")
        print(f"         -> {filepath}  ({len(text):,} chars)\n")
    except Exception as e:
        print(f"         -> {arxiv_id} failed: {e}\n")


def auto_main():
    # date-scoped query for 2025 transformer / LLM papers
    # Fix: Use explicit date range instead of wildcard '*' which causes 500 errors
    query = 'submittedDate:[20240101 TO 20260201]'
    print(f"Querying arxiv with: {query}\n")
    papers = query_arxiv(query=query, max_results=200)

    selected = papers
    print(f"Downloading {len(selected)} papers\n")
    print(selected)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, paper in enumerate(selected, 1):
            futures.append(
                executor.submit(
                    process_paper, paper["arxiv_id"], i, len(selected), paper["title"]
                )
            )
        concurrent.futures.wait(futures)


def main():
    arxiv_ids = open("arxiv_ids.txt").readlines()
    arxiv_ids = [arxiv_id.strip() for arxiv_id in arxiv_ids if arxiv_id.strip()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, arxiv_id in enumerate(arxiv_ids, 1):
            futures.append(executor.submit(process_paper, arxiv_id, i, len(arxiv_ids)))
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    main()
