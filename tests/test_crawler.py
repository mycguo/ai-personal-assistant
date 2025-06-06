from webcrawer import WebCrawler
from unittest.mock import patch


def mock_get(*args, **kwargs):
    class Response:
        def __init__(self, text):
            self.text = text
    html = '<html><body><a href="https://example.com/page">Page</a></body></html>'
    return Response(html)


def test_crawl_returns_root():
    crawler = WebCrawler('https://example.com', max_depth=1)
    with patch('requests.get', side_effect=mock_get):
        urls = crawler.start_crawling('https://example.com')
    assert 'https://example.com' in urls
    assert isinstance(urls, set)

