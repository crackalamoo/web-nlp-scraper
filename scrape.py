import scrapy
import os
import os.path
import re
from scrapy.crawler import CrawlerProcess
from scrapy.http import Request, HtmlResponse
from scrapy.linkextractors import LinkExtractor
from scrapy.utils.project import get_project_settings
from bs4 import BeautifulSoup

class Page(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    # size = scrapy.Field()
    # referer = scrapy.Field()
    # newcookies = scrapy.Field()
    body = scrapy.Field()

def html_to_text(markup, preserve_new_lines=True, strip_tags=['style', 'script', 'code']):
    """
    Based on https://stackoverflow.com/questions/30337528/make-beautifulsoup-handle-line-breaks-as-a-browser-would (Rich - enzedonline)
    """
    NON_BREAKING_ELEMENTS = ['a', 'abbr', 'acronym', 'audio', 'b', 'bdi', 'bdo', 'big', 'button', 
    'canvas', 'cite', 'code', 'data', 'datalist', 'del', 'dfn', 'em', 'embed', 'i', 'iframe', 
    'img', 'input', 'ins', 'kbd', 'label', 'map', 'mark', 'meter', 'noscript', 'object', 'output', 
    'picture', 'progress', 'q', 'ruby', 's', 'samp', 'script', 'select', 'slot', 'small', 'span', 
    'strong', 'sub', 'sup', 'svg', 'template', 'textarea', 'time', 'u', 'tt', 'var', 'video', 'wbr']

    markup = markup.replace('\n',' ').replace('\r\n',' ')
    markup = re.sub(' +', ' ', markup)
    soup = BeautifulSoup(markup, "html.parser")
    for element in soup(strip_tags): element.extract()
    if preserve_new_lines:
        for element in soup.find_all():
            if element.name not in NON_BREAKING_ELEMENTS:
                element.append('\n') if element.name == 'br' else element.append('\n\n')
    text = soup.get_text(" ").strip()
    text = re.sub('\n +', '\n', text)
    text = re.sub(' +\n', '\n', text)
    text = re.sub(' +', ' ', text)
    text = re.sub('\n\n+','\n\n', text)
    return text

class BlogSpider(scrapy.Spider):
    name = 'blog-spider'
    start_urls = ['https://www.harysdalvi.com']

    def __init__(self, **kwargs):
        super(BlogSpider, self).__init__(**kwargs)

    def parse(self, response):
        page = self._get_item(response)
        r = [page]
        r.extend(self._extract_requests(response))
        return r

    def _get_item(self, response):
        item = Page(
            url=response.url,
            body=html_to_text(response.text)
        )
        if isinstance(response, HtmlResponse):
            title = response.xpath("//title/text()").extract()
            if title:
                item['title'] = title[0]
        return item
    
    def _extract_requests(self, response):
        r = []
        if isinstance(response, HtmlResponse):
            r.extend([])
        return r

if __name__ == '__main__':
    settings = get_project_settings()

    if os.path.isfile("out/export.json"):
        os.remove("out/export.json")

    process = CrawlerProcess(
        settings={
            "FEED_FORMAT": "json",
            "FEED_URI": "out/export.json"
        }
    )

    process.crawl(BlogSpider)
    process.start()