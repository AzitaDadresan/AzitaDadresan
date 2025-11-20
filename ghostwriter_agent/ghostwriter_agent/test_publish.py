"""Safe WordPress connectivity tester for GhostWriter.

This module provides a small CLI utility to verify WordPress REST API
credentials using the `WP_SITE`, `WP_USER`, and `WP_PASSWORD` environment
variables. By default it performs an authenticated GET on `/wp-json/wp/v2/users/me`
to validate credentials without creating any posts. Creating a draft post
requires the explicit `--create` flag.

Usage (verify only):
    python -m ghostwriter_agent.test_publish

Create a draft post (explicit):
    python -m ghostwriter_agent.test_publish --create --title "Test" --content "..."

This helper never prints secrets. It will print concise status messages
and, on success, the user id and post id (if a post was created).
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

try:
    # prefer dotenv if available so local `.env` will be read
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _get_wp_site() -> Optional[str]:
    site = os.environ.get("WP_SITE")
    if site:
        return site.rstrip("/")
    return None


def verify_credentials() -> bool:
    """Verify WP credentials by calling `/wp/v2/users/me`.

    Returns True if the request succeeds (HTTP 200). Does not expose secrets.
    """
    site = _get_wp_site()
    if not site:
        logging.error("WP_SITE not set in environment. Cannot verify credentials.")
        return False

    user = os.environ.get("WP_USER")
    pwd = os.environ.get("WP_PASSWORD")
    if not (user and pwd):
        logging.error("WP_USER or WP_PASSWORD not set in environment. Cannot verify credentials.")
        return False

    url = f"{site}/wp-json/wp/v2/users/me"
    try:
        resp = requests.get(url, auth=HTTPBasicAuth(user, pwd), timeout=15)
    except Exception as e:
        logging.error("Request failed: %s", e)
        return False

    if resp.status_code == 200:
        try:
            data = resp.json()
            uid = data.get("id")
            name = data.get("name") or data.get("slug")
            logging.info("Credentials verified: user id=%s, name=%s", uid, name)
        except Exception:
            logging.info("Credentials verified (HTTP 200).")
        return True

    if resp.status_code in (401, 403):
        logging.error("Authentication failed (HTTP %s). Check credentials or use an Application Password.", resp.status_code)
    else:
        logging.error("Unexpected response from server (HTTP %s).", resp.status_code)
    return False


def create_draft_post(title: str, content: str) -> Optional[int]:
    """Create a draft post. Returns the post id on success, otherwise None.

    This function should only be used with explicit user consent. The CLI
    requires the `--create` flag to call this.
    """
    site = _get_wp_site()
    if not site:
        logging.error("WP_SITE not set in environment. Cannot create post.")
        return None

    user = os.environ.get("WP_USER")
    pwd = os.environ.get("WP_PASSWORD")
    if not (user and pwd):
        logging.error("WP_USER or WP_PASSWORD not set in environment. Cannot create post.")
        return None

    url = f"{site}/wp-json/wp/v2/posts"
    payload = {"title": title, "content": content, "status": "draft"}
    try:
        resp = requests.post(url, auth=HTTPBasicAuth(user, pwd), json=payload, timeout=20)
    except Exception as e:
        logging.error("Request failed: %s", e)
        return None

    if resp.status_code in (200, 201):
        try:
            data = resp.json()
            post_id = data.get("id")
            link = data.get("link")
            logging.info("Draft created: id=%s, link=%s", post_id, link)
            return post_id
        except Exception:
            logging.info("Draft created (HTTP %s).", resp.status_code)
            return None

    logging.error("Failed to create draft (HTTP %s): %s", resp.status_code, resp.text[:400])
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify WordPress credentials and optionally create a draft post.")
    parser.add_argument("--create", action="store_true", help="Create a draft post (requires --title and --content).")
    parser.add_argument("--title", type=str, default="Test draft from GhostWriter", help="Title for the draft post.")
    parser.add_argument("--content", type=str, default="This is a safe test draft created by GhostWriter. Delete if unnecessary.", help="Content for the draft post.")
    args = parser.parse_args(argv)

    ok = verify_credentials()
    if not ok:
        logging.error("Verification failed. Aborting.")
        return 2

    if args.create:
        logging.info("Creating draft post (status=draft)...")
        pid = create_draft_post(args.title, args.content)
        if pid:
            logging.info("Draft creation succeeded: id=%s", pid)
            return 0
        logging.error("Draft creation failed.")
        return 3

    logging.info("Verification only; no posts created. Use --create to create a draft.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
