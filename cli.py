import sys
import argparse
import os
import traceback
import json
import pickle
import readline

from scrape import output_scrape
from models import topic_modeling

def get_parser():
    parser = argparse.ArgumentParser(
        prog='web-nlp-scraper-cli',
        description='Web NLP scraper CLI'
    )

    subparsers = parser.add_subparsers(title='command', metavar='command', dest='command', help='Command to run.')
    parser_quit = subparsers.add_parser('quit', help='Exit the CLI (can also use q)')
    parser_urls = subparsers.add_parser('urls', help='List all URLs.')

    parser_load = subparsers.add_parser('load', help='Load data from file.')
    parser_load.add_argument('file', default='out/export.json', nargs='?', help='File to read from. Default: %(default)s')

    parser_scrape = subparsers.add_parser('scrape', help='Scrape website data and output to disk.')
    parser_scrape.add_argument('url', help='URL to scrape. Examples: example.com or https://www.example.com')
    parser_scrape.add_argument('--allow', nargs='*', help='List of RegEx expressions for allowed sub-URLs. Example: --allow sub\/* to allow all https://www.example.com/sub/*')
    parser_scrape.add_argument('--deny', nargs='*', help='List of RegEx expressions for disallowed sub-URLs. Example: --deny sub\/* to disallow all https://www.example.com/sub/*')
    parser_scrape.add_argument('--keep-langs', nargs='*', help='For a multilingual site, only tags with no lang or the given languages will be scraped. Example: --keep-langs en es will keep all English- and Spanish-language content, and ignore elements with any other lang tag. Default: use content in all languages.')
    parser_scrape.add_argument('--feed-uri', default='out/export.json', help='File to output scraped data. Default: %(default)s')
    parser_scrape.add_argument('--feed-format', default='json', choices=['json'], help='Format to output scraped data. Default: %(default)s')

    parser_topics = subparsers.add_parser('topics', help='Topic modeling')
    parser_topics.add_argument('n_topics', nargs='?', type=int, default=5, help='Number of topics')

    return parser

def parse_command(cmd, parser, data_dict):
    args = parser.parse_args(cmd.split())
    if args.command == 'load':
        if args.file.endswith('.json'):
            with open(args.file, 'r') as f:
                loaded = json.load(f)
        else:
            with open(args.file, 'rb') as f:
                loaded = pickle.load(f, encoding='utf-8')
        data_dict.clear()
        data_dict += loaded
        print(f"Successfully loaded {args.file}.")
    elif args.command == 'urls':
        urls = []
        for page in data_dict:
            urls.append(page['url'])
        print(', '.join(urls))
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
        topics = topic_modeling(data_dict, n_topics=args.n_topics)
        for i, topic in topics.iterrows():
            print(topic.sort_values(ascending=False).head(10))

def main():
    parser = get_parser()
    data_dict = []
    if os.path.isfile('out/export.json'):
        with open('out/export.json', 'r') as f:
            data_dict = json.load(f)
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