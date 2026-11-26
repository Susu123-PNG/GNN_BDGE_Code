"""Download public data archives from their original publishers."""
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URLS = {
    'onet31': 'https://www.onetcenter.org/dl_files/database/db_31_0_csv.zip',
    'esco_onet_crosswalk': 'https://esco.ec.europa.eu/system/files/2023-08/ONET_%28Occupations%29_0_updated.csv',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=list(URLS), required=True)
    parser.add_argument('--directory', default='data/input')
    args = parser.parse_args()
    folder = Path(args.directory)
    folder.mkdir(parents=True, exist_ok=True)
    url = URLS[args.target]
    file = folder / ('onet31.csv.zip' if args.target == 'onet31' else 'esco_onet_crosswalk.csv')
    with urllib.request.urlopen(url, timeout=120) as response, file.open('wb') as output:
        while True:
            chunk = response.read(1024*1024)
            if not chunk:
                break
            output.write(chunk)
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    (folder / (file.name + '.source.json')).write_text(json.dumps({
        'downloaded_utc': datetime.now(timezone.utc).isoformat(),
        'url': url, 'sha256': digest, 'bytes': file.stat().st_size
    }, indent=2))
    print(file, digest)


if __name__ == '__main__':
    main()
