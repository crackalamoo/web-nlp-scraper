import json

def write_md(scrape_json='out/export.json', out='out/out.md', skip_titles=None):
    with open(scrape_json, 'r') as f:
        articles = json.load(f)

    t = ''
    for article in articles:
        if article['title'] in skip_titles:
            continue
        t += '# ' + article['body'] + '\n\n'

    with open (out, 'w') as f:
        f.write(t)

if __name__ == '__main__':
    write_md(skip_titles=set(['Harys Dalvi']))