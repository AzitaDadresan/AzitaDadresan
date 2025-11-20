import requests
from urllib.parse import urljoin

def check_url(url):
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        return r
    except:
        return None

def is_wordpress(site):
    site = site.rstrip("/")

    signals = {
        "wp_json": False,
        "wp_content": False,
        "wp_login": False,
        "meta_generator": False,
        "headers_powered": False
    }

    # 1. Check wp-json (strongest)
    r = check_url(urljoin(site, "/wp-json/"))
    if r and r.status_code == 200:
        try:
            data = r.json()
            if isinstance(data, dict):  # valid json
                signals["wp_json"] = True
        except:
            pass

    # 2. Check wp-content reference on homepage
    r = check_url(site)
    if r:
        html = r.text.lower()
        if "wp-content" in html or "wp-includes" in html:
            signals["wp_content"] = True

        # meta generator tag
        if "generator\" content=\"wordpress" in html:
            signals["meta_generator"] = True

    # 3. Check login page existence
    r = check_url(urljoin(site, "/wp-login.php"))
    if r and r.status_code in [200, 302]:
        signals["wp_login"] = True

    # 4. Check headers
    if r and "x-powered-by" in r.headers:
        if "wordpress" in r.headers["x-powered-by"].lower():
            signals["headers_powered"] = True

    # Simple scoring
    score = sum(signals.values())

    return {
        "is_wordpress": score >= 2,  # threshold
        "score": score,
        "signals": signals
    }

# Test ----------------------------------------
sites = [
    "https://zeatz.in",
    "http://rohitconsultants.com",
    "https://publicationsensemble.com/",
    "https://luxeensemble.com"
]

for s in sites:
    print(s, is_wordpress(s))
