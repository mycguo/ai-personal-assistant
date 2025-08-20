#!/usr/bin/env python3
"""
Enhanced URL fetcher that handles compression issues properly.
Specifically addresses the garbled text problem with https://helpme.tebra.com/
"""

import requests
from bs4 import BeautifulSoup
import sys

def fetch_url_content(url, disable_compression=True):
    """
    Fetch URL content with proper encoding and compression handling.
    
    Args:
        url (str): The URL to fetch
        disable_compression (bool): Whether to disable compression to avoid encoding issues
    
    Returns:
        tuple: (success, content_or_error)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    if disable_compression:
        headers["Accept-Encoding"] = "identity"
    
    try:
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        response.raise_for_status()
        
        # Handle encoding explicitly
        if response.encoding is None:
            response.encoding = 'utf-8'
        
        return True, response.text
        
    except requests.exceptions.RequestException as err:
        return False, f"Request error: {err}"
    except UnicodeDecodeError as err:
        return False, f"Encoding error: {err}"

def extract_readable_content(html_content):
    """
    Extract readable content from HTML, similar to WebFetch functionality.
    
    Args:
        html_content (str): Raw HTML content
    
    Returns:
        str: Cleaned readable content
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as err:
        return f"Content extraction error: {err}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python url_fetcher_test.py <URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    print(f"Testing URL: {url}")
    
    # Test with compression disabled
    print("\n--- Testing with compression disabled ---")
    success, content = fetch_url_content(url, disable_compression=True)
    
    if success:
        print(f"✅ Success! Content length: {len(content)} characters")
        print("\nFirst 500 characters:")
        print(content[:500])
        
        print("\n--- Extracted readable content (first 1000 chars) ---")
        readable = extract_readable_content(content)
        print(readable[:1000])
    else:
        print(f"❌ Failed: {content}")
    
    # Test with compression enabled for comparison
    print("\n--- Testing with compression enabled ---")
    success2, content2 = fetch_url_content(url, disable_compression=False)
    
    if success2:
        print(f"✅ Success with compression! Content length: {len(content2)} characters")
        print("First 500 characters:")
        print(content2[:500])
    else:
        print(f"❌ Failed with compression: {content2}")