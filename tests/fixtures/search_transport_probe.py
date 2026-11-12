"""Minimal executable for qualifying the native search transport bundle."""
import json
import sys

from search_http import download_search_html

if __name__ == '__main__':
    try:
        status, body = download_search_html(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))
        print(json.dumps({'status': status, 'body': body.decode()}))
    except Exception as error:
        print(json.dumps({'error': type(error).__name__, 'message': str(error)}))
        sys.exit(1)
