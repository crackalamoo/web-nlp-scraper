import scrapy
import re
from scrapy.crawler import CrawlerProcess
from scrapy.http import Request, HtmlResponse
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
import urllib.parse
import json
import pickle

class Page(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    # size = scrapy.Field()
    # referer = scrapy.Field()
    # newcookies = scrapy.Field()
    body = scrapy.Field()

def html_to_text(markup, preserve_new_lines=True, remove_hidden=True, keep_langs=None, strip_tags=None):
    """
    Based on https://stackoverflow.com/questions/30337528/make-beautifulsoup-handle-line-breaks-as-a-browser-would (Rich - enzedonline)
    """
    NON_BREAKING_ELEMENTS = set(['a', 'abbr', 'acronym', 'audio', 'b', 'bdi', 'bdo', 'big', 'button', 
    'canvas', 'cite', 'code', 'data', 'datalist', 'del', 'dfn', 'em', 'embed', 'i', 'iframe', 
    'img', 'input', 'ins', 'kbd', 'label', 'map', 'mark', 'meter', 'noscript', 'object', 'output', 
    'picture', 'progress', 'q', 'ruby', 's', 'samp', 'script', 'select', 'slot', 'small', 'span', 
    'strong', 'sub', 'sup', 'svg', 'template', 'textarea', 'time', 'u', 'tt', 'var', 'video', 'wbr'])

    if strip_tags is None:
        strip_tags = ['style', 'script', 'wbr']
    if keep_langs is not None and type(keep_langs) != set:
        keep_langs = set(keep_langs)
    
    markup = markup.replace('\n',' ').replace('\r\n',' ')
    markup = re.sub(r' +', ' ', markup)
    soup = BeautifulSoup(markup, 'html.parser')
    

    for element in soup(strip_tags):
        element.extract()
    for element in soup.find_all():
        if remove_hidden:
            try:
                if 'display:none' in element['style']:
                    element.extract()
            except KeyError:
                pass
        if keep_langs is not None:
            def lang_included(lang):
                if lang is None:
                    return True
                if element['lang'] in keep_langs or element['lang'][:2] in keep_langs and element[2:3] == '-':
                    return True
                return False
            try:
                if not lang_included(element['lang']):
                    element.extract()
            except KeyError:
                pass
    if preserve_new_lines:
        for element in soup.find_all():
            if element.name not in NON_BREAKING_ELEMENTS:
                element.append('\n') if element.name == 'br' else element.append('\n\n')
    text = soup.get_text(' ').strip()
    text = re.sub(r'\n +', '\n', text)
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n+', '\n\n', text)
    return text

class BlogSpider(scrapy.Spider):
    name = 'blog-spider'

    def __init__(self, **kwargs):
        super(BlogSpider, self).__init__(**kwargs)

        url = kwargs.get('url') or kwargs.get('domain') or 'https://www.harysdalvi.com'
        if not url.startswith('http://') and not url.startswith('https://'):
            url = f'http://{url}'
        self.url = url

        hostname = re.sub(r'^www\.', '', urllib.parse.urlsplit(url).hostname)

        allow = kwargs.get('allow')
        deny = kwargs.get('deny')
        if deny is not None:
            for i in range(len(deny)):
                if not deny[i].startswith(self.url):
                    deny[i] = f'https?:\/\/(www\.)?{hostname}\/' + deny[i]
        if allow is not None:
            for i in range(len(allow)):
                if not allow[i].startswith(self.url):
                    allow[i] = f'https?:\/\/(www\.)?{hostname}\/' + allow[i]
        
        self.link_extractor = LinkExtractor(allow_domains=[hostname], allow=allow, deny=deny)

        self.keep_langs = kwargs.get('keep_langs')

    def start_requests(self):
        return [Request(self.url, callback=self.parse, dont_filter=True)]

    def parse(self, response):
        page = self._get_item(response)
        r = [page]
        r.extend(self._extract_requests(response))
        return r

    def _get_item(self, response):
        item = Page(
            url=response.url,
            body=html_to_text(response.text, keep_langs=self.keep_langs)
        )
        if isinstance(response, HtmlResponse):
            title = response.xpath('//title/text()').extract()
            if title:
                item['title'] = title[0]
        return item
    
    def _extract_requests(self, response):
        r = []
        if isinstance(response, HtmlResponse):
            links = self.link_extractor.extract_links(response)
            r.extend(Request(x.url, callback=self.parse) for x in links)
        return r

def output_scrape(**kwargs):
    feed_uri = kwargs.get('feed_uri', 'out/export.json')

    process = CrawlerProcess(
        settings={
            'FEEDS': {
                feed_uri: {
                    'format': kwargs.get('feed_format', 'json'),
                    'uri': feed_uri,
                    'encoding': 'utf8',
                    'overwrite': True
                }
            }
        }
    )

    process.crawl(BlogSpider, **kwargs)
    process.start()
    process.join()

def load_scrape(file):
    if file.endswith('.json'):
        with open(file, 'r') as f:
            loaded = json.load(f)
    else:
        with open(file, 'rb') as f:
            loaded = pickle.load(f, encoding='utf-8')
    return loaded

if __name__ == '__main__':
    output_scrape(url='www.harysdalvi.com',
                  allow=[r'blog\/*'],
                  deny=[r'sub\/*', 'calcalc', 'names', 'ahsgrades'],
                  keep_langs=['en'])