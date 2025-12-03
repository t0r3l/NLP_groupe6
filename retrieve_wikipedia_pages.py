#!/usr/bin/env python3
"""
Script to retrieve Wikipedia pages content from URLs in a CSV file.
Uses the French Wikipedia API to fetch page content.
"""

import csv
import json
import os
import re
import time
from urllib.parse import unquote, urlparse

import requests

# Wikipedia API endpoint for French Wikipedia
WIKIPEDIA_API_URL = "https://fr.wikipedia.org/w/api.php"

# User-Agent is required by Wikipedia API policy
# https://meta.wikimedia.org/wiki/User-Agent_policy
HEADERS = {
    "User-Agent": "AfricanCivilizationsBot/1.0 (NLP Project; contact@example.com) Python/requests"
}

# Output directory for retrieved pages
OUTPUT_DIR = "wikipedia_pages"


def extract_page_title_from_url(url: str) -> str | None:
    """Extract the Wikipedia page title from a URL."""
    if not url or not url.strip():
        return None
    
    try:
        parsed = urlparse(url)
        path = parsed.path
        
        # Extract title from /wiki/Title pattern
        if "/wiki/" in path:
            title = path.split("/wiki/")[-1]
            # URL decode the title (handles special characters like é, à, etc.)
            title = unquote(title)
            if title:
                return title
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
    
    return None


def get_wikipedia_page_content(title: str) -> dict | None:
    """
    Retrieve Wikipedia page content using the API.
    Returns a dict with title, content, and metadata.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info|categories",
        "exintro": False,  # Get full content, not just intro
        "explaintext": True,  # Get plain text instead of HTML
        "format": "json",
        "inprop": "url"
    }
    
    try:
        response = requests.get(WIKIPEDIA_API_URL, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page_data in pages.items():
            if page_id == "-1":
                # Page not found
                print(f"  Page not found: {title}")
                return None
            
            return {
                "page_id": page_id,
                "title": page_data.get("title", title),
                "content": page_data.get("extract", ""),
                "url": page_data.get("fullurl", ""),
                "categories": [cat.get("title", "") for cat in page_data.get("categories", [])]
            }
    
    except requests.RequestException as e:
        print(f"  Error fetching page {title}: {e}")
        return None
    
    return None


def sanitize_filename(title: str) -> str:
    """Convert a Wikipedia title to a safe filename."""
    # Replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', title)
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def main():
    csv_path = "data/raw/civilisations_afrique_precoloniale.csv"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    entries = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    
    print(f"Found {len(entries)} entries in CSV")
    
    # Track results
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    # Process each entry
    for i, entry in enumerate(entries):
        entity_name = entry.get("entite", "Unknown")
        url = entry.get("url", "")
        
        print(f"\n[{i+1}/{len(entries)}] Processing: {entity_name}")
        
        # Extract page title from URL
        title = extract_page_title_from_url(url)
        
        if not title:
            print(f"  Skipping - could not extract title from URL: {url}")
            skipped += 1
            results.append({
                "entity": entity_name,
                "url": url,
                "status": "skipped",
                "reason": "Could not extract title from URL"
            })
            continue
        
        print(f"  Fetching page: {title}")
        
        # Fetch page content
        page_data = get_wikipedia_page_content(title)
        
        if page_data and page_data.get("content"):
            # Save content to file
            filename = sanitize_filename(entity_name) + ".txt"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Entity: {entity_name}\n")
                f.write(f"Wikipedia Title: {page_data['title']}\n")
                f.write(f"URL: {page_data['url']}\n")
                f.write(f"Region: {entry.get('region', 'N/A')}\n")
                f.write(f"Period: {entry.get('debut', 'N/A')} - {entry.get('fin', 'N/A')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(page_data["content"])
            
            print(f"  Saved to: {filepath}")
            successful += 1
            results.append({
                "entity": entity_name,
                "url": url,
                "wikipedia_title": page_data["title"],
                "status": "success",
                "output_file": filepath
            })
        else:
            print(f"  Failed to retrieve content")
            failed += 1
            results.append({
                "entity": entity_name,
                "url": url,
                "status": "failed",
                "reason": "Could not retrieve page content"
            })
        
        # Be nice to Wikipedia servers - add small delay between requests
        time.sleep(0.5)
    
    # Save summary report
    report_path = os.path.join(OUTPUT_DIR, "_retrieval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(entries),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RETRIEVAL SUMMARY")
    print("=" * 60)
    print(f"Total entries:  {len(entries)}")
    print(f"Successful:     {successful}")
    print(f"Failed:         {failed}")
    print(f"Skipped:        {skipped}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

