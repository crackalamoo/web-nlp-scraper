import sys
import argparse
import os
import traceback
import json
import pickle

def get_parser():
    parser = argparse.ArgumentParser(
        prog='web-nlp-scraper-cli',
        description='Web NLP scraper CLI'
    )

    subparsers = parser.add_subparsers(title='command', metavar='command', dest='command', help='Command to run.')
    parser_urls = subparsers.add_parser('urls', help='list all urls')
    parser_load = subparsers.add_parser('load', help='load data from file')
    parser_load.add_argument('-f', '--file', default='out/export.pkl', help='file to read from (default: %(default)s)')
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

def main():
    parser = get_parser()
    data_dict = []
    if os.path.isfile('out/export.json'):
        with open('out/export.json', 'r') as f:
            data_dict = json.load(f)
        print("Successfully loaded out/export.json.")

    while True:
        try:
            cmd = input('>>> ')
            parse_command(cmd, parser, data_dict)
        except Exception:
            sys.stderr.write(traceback.format_exc())
        except SystemExit:
            pass

if __name__ == '__main__':
    main()