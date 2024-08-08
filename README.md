# Web NLP Scraper

A command line tool to quickly run natural language processing (NLP) algorithms on any website.
Ideal for understanding the language trends of a blog, or comparing two blogs.

Features:

* **Scraping.** Scrape and clean pages on a website using `scrapy`.
* **Comparison.** Run a variety of NLP algorithms to compare differences in style and subject matter between two websites.
* **Topic modeling.** Use linear discriminant analysis to determine the most common topics discussed on a website.
* **Document similarity.** Use term frequency-inverse document frequency (TF-IDF) to determine the most and least similar pages within a website or between two websites.
* **Top words.** Get the most disproportionately common words of each page on a website, or each of two websites.
* **Named entity recognition.** Get a list of the most disproportionately named entities in a page of a website or between websites.
* **Style comparison.** Get the most disproportionately common *style* words (non-content words) across pages or between two websites.
* **Classifier.** Train a transformer model to distinguish between sentences on two websites.