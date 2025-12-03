"""
Script to fetch 20 Wikipedia pages on various subjects and save them locally.
Uses the Wikipedia API directly via requests.
"""

import requests
import json
import os
from pathlib import Path

# Wikipedia API endpoint
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

# List of 20 diverse topics to fetch
TOPICS = [
    "Artificial intelligence",
    "Climate change",
    "Quantum computing",
    "Renaissance art",
    "Ancient Rome",
    "Machine learning",
    "Black holes",
    "French Revolution",
    "DNA",
    "Mozart",
    "World War II",
    "Photosynthesis",
    "Cryptocurrency",
    "Greek mythology",
    "Impressionism",
    "Theory of relativity",
    "Amazon rainforest",
    "Industrial Revolution",
    "Human rights",
    "Solar system"
]


def fetch_wikipedia_page(title: str) -> dict | None:
    """
    Fetch a Wikipedia page by title using the API.
    Returns the page content and metadata.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info",
        "explaintext": True,  # Get plain text instead of HTML
        "format": "json",
        "redirects": 1  # Follow redirects
    }
    
    try:
        response = requests.get(WIKI_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id != "-1":  # -1 means page not found
                return {
                    "title": page_data.get("title", title),
                    "page_id": page_id,
                    "content": page_data.get("extract", ""),
                    "length": page_data.get("length", 0)
                }
        return None
    except requests.RequestException as e:
        print(f"Error fetching '{title}': {e}")
        return None


def save_page(page_data: dict, output_dir: Path) -> None:
    """Save a Wikipedia page to a JSON file."""
    filename = page_data["title"].replace("/", "_").replace(" ", "_") + ".json"
    filepath = output_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved: {page_data['title']} ({len(page_data['content'])} chars)")


def main():
    # Create output directory
    output_dir = Path(__file__).parent / "wikipedia_pages"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Fetching {len(TOPICS)} Wikipedia pages...")
    print(f"Output directory: {output_dir}\n")
    
    successful = 0
    failed = []
    
    for topic in TOPICS:
        print(f"Fetching: {topic}...", end=" ")
        page_data = fetch_wikipedia_page(topic)
        
        if page_data:
            save_page(page_data, output_dir)
            successful += 1
        else:
            print(f"✗ Failed to fetch: {topic}")
            failed.append(topic)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Summary: {successful}/{len(TOPICS)} pages fetched successfully")
    if failed:
        print(f"Failed topics: {', '.join(failed)}")
    print(f"Pages saved to: {output_dir}")


if __name__ == "__main__":
    main()

