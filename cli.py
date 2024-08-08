import sys
import argparse
import os
import traceback
import json
import numpy as np
import pandas as pd
import pickle
import readline

from scrape import output_scrape, load_scrape
from models import topic_modeling, get_top_words, get_similarities, named_entity_recognition, stopwords_tf_idf, bert_classifier

def get_parser():
    parser = argparse.ArgumentParser(
        prog='web-nlp-scraper-cli',
        description='Web NLP scraper CLI'
    )

    subparsers = parser.add_subparsers(title='command', metavar='command', dest='command', help='Command to run.')
    subparsers.add_parser('quit', help='Exit the CLI (can also use q)')
    subparsers.add_parser('urls', help='List URLs of all pages loaded in memory.')
    subparsers.add_parser('titles', help='List titles of all pages loaded in memory, falling back on URL if no title.')

    parser_load = subparsers.add_parser('load', help='Load data from a file created by scrape.')
    parser_load.add_argument('file', default='out/export.json', nargs='?', help='File to read from. Default: %(default)s')
    parser_compare = subparsers.add_parser('compare', help='Load data from an additional file, for comparison with the one loaded by load.')
    parser_compare.add_argument('file', default='out/compare.json', nargs='?', help='File to read from. Default: %(default)s')

    parser_scrape = subparsers.add_parser('scrape', help='Scrape website data and output to disk.')
    parser_scrape.add_argument('url', help='URL to scrape. Examples: example.com or https://www.example.com')
    parser_scrape.add_argument('--allow', nargs='*', help='List of RegEx expressions for allowed sub-URLs. Example: --allow sub\/* to allow all https://www.example.com/sub/*')
    parser_scrape.add_argument('--deny', nargs='*', help='List of RegEx expressions for disallowed sub-URLs. Example: --deny sub\/* to disallow all https://www.example.com/sub/*')
    parser_scrape.add_argument('--keep-langs', nargs='*', help='For a multilingual site, only tags with no lang or the given languages will be scraped. Example: --keep-langs en es will keep all English- and Spanish-language content, and ignore elements with any other lang tag. Default: use content in all languages.')
    parser_scrape.add_argument('--feed-uri', default='out/export.json', help='File to output scraped data. Default: %(default)s')
    parser_scrape.add_argument('--feed-format', default='json', choices=['json'], help='Format to output scraped data. Default: %(default)s')

    parser_topics = subparsers.add_parser('topics', help='Topic modeling')
    parser_topics.add_argument('n_topics', nargs='?', type=int, default=5, help='Number of topics. Default: %(default)s')
    parser_topics.add_argument('-c', '--compare', action='store_true', help='Include pages from the comparison website.')

    parser_top_words = subparsers.add_parser('top-words', help='Get the most disproportionately common words for each page (by TF-IDF).')
    parser_top_words.add_argument('n_words', nargs='?', type=int, default=5, help='Number of words to display per topic. Default: %(default)s')
    parser_top_words.add_argument('-c', '--compare', action='store_true', help='Compare words in the loaded website and the comparison website, rather than between pages in the loaded website.')

    parser_sims = subparsers.add_parser('sims', help='Document similarity using TF-IDF')
    parser_sims.add_argument('-c', '--compare', action='store_true', help='Include pages from the comparison website.')

    parser_entities = subparsers.add_parser('entities', help='Named entity recognition')
    parser_entities.add_argument('n_entities', nargs='?', type=int, default=15, help='Number of entities to display. Default: %(default)s')
    parser_entities.add_argument('-c', '--compare', action='store_true', help='Display top named entities for both the loaded website and the comparison website.')
    parser_entities.add_argument('--spacy-pipeline', default='en_core_web_sm', help='Spacy pipeline. Default: %(default)s')

    parser_style = subparsers.add_parser('style', help='Style analysis: get top stopwords, indicating disproportionately used stylistic words')
    parser_style.add_argument('n_words', nargs='?', type=int, default=5, help='Number of words to display per topic. Default: %(default)s')
    parser_style.add_argument('-c', '--compare', action='store_true', help='Display top named entities for both the loaded website and the comparison website.')
    parser_style.add_argument('--spacy-pipeline', default='en_core_web_sm', help='Spacy pipeline. Default: %(default)s')

    parser_classifier = subparsers.add_parser('classifier', help='Train a BERT classifier between the loaded and comparison websites')

    return parser

def parse_command(cmd, parser, data_dict):
    args = parser.parse_args(cmd.split())
    if args.command == 'load':
        loaded = load_scrape(args.file)
        data_dict['page_list'] = loaded
        print(f"Successfully loaded {args.file}.")
    elif args.command == 'compare':
        loaded = load_scrape(args.file)
        data_dict['compare_page_list'] = loaded
        print(f"Successfully loaded {args.file} for comparison.")
    elif args.command == 'urls':
        urls = []
        for page in data_dict['page_list']:
            urls.append(page['url'])
        print(', '.join(urls))
    elif args.command == 'titles':
        titles = []
        for page in data_dict['page_list']:
            titles.append(page['title'] if 'title' in page else page['url'])
        print(', '.join(titles))
    if args.command == 'scrape':
        output_scrape(
            url=args.url,
            allow=args.allow,
            deny=args.deny,
            keep_langs=args.keep_langs,
            feed_uri=args.feed_uri,
            feed_format=args.feed_format
        )
    elif args.command == 'topics':
        if args.compare:
            topics = topic_modeling(data_dict['page_list'], compare_list=data_dict['compare_page_list'], n_topics=args.n_topics)
        else:
            topics = topic_modeling(data_dict['page_list'], n_topics=args.n_topics)
        for i, topic in topics.iterrows():
            print(topic.sort_values(ascending=False).head(10))
    elif args.command == 'top-words':
        if args.compare:
            df = get_top_words(data_dict['page_list'], compare_list=data_dict['compare_page_list'])
        else:
            df = get_top_words(data_dict['page_list'])
        for series_name, series in df.items():
            print(series_name)
            print(series.sort_values(ascending=False).head(args.n_words))
    elif args.command == 'sims':
        if args.compare:
            sims, sims_list, diff_pairs = get_similarities(data_dict['page_list'], compare_list=data_dict['compare_page_list'])
        else:
            sims, sims_list = get_similarities(data_dict['page_list'])
        print(sims_list.sort_values(ascending=False).head())
        print(sims_list.sort_values(ascending=False).tail())
        if args.compare:
            print(diff_pairs.sort_values(ascending=False).head())
            print(diff_pairs.sort_values(ascending=False).tail())
    elif args.command == 'entities':
        entity_counts = named_entity_recognition(data_dict['page_list'], args.spacy_pipeline)
        if args.compare:
            print("Loaded website:")
        for entity in entity_counts.most_common(args.n_entities):
            print(f'{entity[0][0]} ({entity[0][1]}): {entity[1]}')
        if args.compare:
            print("\nComparison website:")
            entity_counts = named_entity_recognition(data_dict['compare_page_list'], args.spacy_pipeline)
            for entity in entity_counts.most_common(args.n_entities):
                print(f'{entity[0][0]} ({entity[0][1]}): {entity[1]}')
    elif args.command == 'style':
        if args.compare:
            df = stopwords_tf_idf(data_dict['page_list'], compare_list=data_dict['compare_page_list'], remove_stop=False, pipeline=args.spacy_pipeline)
        else:
            df = stopwords_tf_idf(data_dict['page_list'], remove_stop=False, pipeline=args.spacy_pipeline)
        for series_name, series in df.items():
            print(series_name)
            print(series.sort_values(ascending=False).head(args.n_words))
    elif args.command == 'classifier':
        if 'compare_page_list' not in data_dict:
            print("Must load another website for comparison")
            return
        res = bert_classifier(data_dict['page_list'], data_dict['compare_page_list'])

def main():
    parser = get_parser()
    data_dict = {'page_list':[]}
    if os.path.isfile('out/export.json'):
        with open('out/export.json', 'r') as f:
            data_dict['page_list'] = json.load(f)
        print("Successfully loaded out/export.json.")

    while True:
        # scrape harysdalvi.com --deny sub\/* calcalc ahsgrades names --keep-langs en
        try:
            cmd = input('>>> ')
            parse_command(cmd, parser, data_dict)
        except Exception:
            sys.stderr.write(traceback.format_exc())
        except SystemExit: # for argparse
            pass
        try:
            if cmd.split()[0] == 'quit' or cmd.split()[0] == 'q':
                sys.exit(0)
        except IndexError:
            pass

if __name__ == '__main__':
    main()