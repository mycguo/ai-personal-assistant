import requests
import argparse
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import streamlit as st

class WebCrawler:
    def __init__(self, url, max_depth):
        self.url = url
        self.subdomains = set()
        self.max_depth = max_depth

    def start_crawling(self, url):
        return self.crawl(url, depth=0)

    def crawl(self, url, depth):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Encoding": "identity"  # Disable compression to avoid encoding issues
        }
        self.subdomains.add(url)
       
        try:
            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Handle encoding explicitly
            if response.encoding is None:
                response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as err:
            print(f"[-] An error occurred: {err}")
            return self.subdomains
        except UnicodeDecodeError as err:
            print(f"[-] Encoding error occurred: {err}")
            return self.subdomains

        subdomain_query = fr"{url}([a-zA-Z0-9.-]+)"

        # repeat this maxDepth times 
        if depth < self.max_depth: 
            depth = depth + 1
            for link in soup.find_all('a'):
                if 'href' in link.attrs:  
                    href = link.get('href')
                    if href:
                        if re.match(subdomain_query, href) and href not in self.subdomains:
                            self.subdomains.add(href)
                        if href.startswith("/"):            
                            full_link = urljoin(url, href)
                            if full_link != url and full_link not in self.subdomains:
                                self.subdomains.add(full_link)
                                self.crawl(full_link, depth)


        return self.subdomains

    def print_results(self):
        st.write("All the URLs porcessed")
        if self.subdomains:
            for subdomain in self.subdomains:
                print(f"[+]: {subdomain}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', dest='url', help="Specify the URL, provide it along http/https", required=True)
    parser.add_argument('-d', '--depth', dest='depth', type=int, default=1, help="Specify the recursion depth limit")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    web_crawler = WebCrawler(args.url, args.depth)
    web_crawler.start_crawling(args.url)
    web_crawler.print_results()