import urllib.request
import re

url = 'https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=23692892'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        html = r.read().decode('utf-8', errors='ignore')
        for m in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.DOTALL):
            href, text = m.group(1), re.sub(r'<[^>]+>', '', m.group(2)).strip()
            if any(k in (href + ' ' + text).lower() for k in ['radiomic', 'feature', '.csv', 'clinical', 'survival', 'labels', 'download']):
                print(f'{text} -> {href}')
except Exception as e:
    print('Error:', e)
