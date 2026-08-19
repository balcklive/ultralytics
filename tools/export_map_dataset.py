# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Download mxdzlk maps with monster labels as a YOLO dataset.

Examples:
    python tools/export_map_dataset.py --output datasets/mxdzlk_cmsc --version CMSC
    python tools/export_map_dataset.py --output datasets/mxdzlk_one --map-id 40001 --version CMSC
    python tools/export_map_dataset.py --output datasets/mxdzlk_sample --limit 5 --overwrite
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
import yaml
from PIL import Image

DEFAULT_INDEX_URL = "https://mxdzlk.com/map/"
DEFAULT_MARKER_WIDTH = 42
DEFAULT_MARKER_HEIGHT = 42
DEFAULT_BACKGROUND = (238, 242, 241, 255)
MAP_LINK_RE = re.compile(r"/map/(\d+)/?$")
PAGE_COUNT_RE = re.compile(r"第\s*\d+\s*/\s*(\d+)\s*页")
MAP_SECTION_RE = re.compile(r"<section\b[^>]*data-map-visualization[^>]*>([\s\S]*?)</section>", re.IGNORECASE)
MONSTER_RE = re.compile(
    r'<a\b(?=[^>]*class=["\'][^"\']*map-visualization__marker--monster[^"\']*["\'])'
    r"[^>]*>([\s\S]*?)</a>",
    re.IGNORECASE,
)
ATTR_RE = re.compile(r"([\w-]+)\s*=\s*([\"'])(.*?)\2", re.DOTALL)


@dataclass(frozen=True)
class Monster:
    """A monster marker in render-space pixels."""

    name: str
    icon_url: str
    x: float
    y: float


@dataclass(frozen=True)
class MapPage:
    """Map metadata and its monster markers."""

    map_id: str
    name: str
    image_url: str
    width: int
    height: int
    monsters: tuple[Monster, ...]


def set_version(url: str, version: str | None) -> str:
    """Add or replace the data-version query parameter on a URL."""
    if not version:
        return url
    parsed = urlparse(url)
    query = [(key, value) for key, value in parse_qsl(parsed.query) if key not in {"v", "sv"}]
    query.append(("v", version))
    return urlunparse(parsed._replace(query=urlencode(query)))


def request(session: requests.Session, url: str, timeout: float) -> requests.Response:
    """Request a URL with bounded retry-after handling for rate limits."""
    for attempt in range(4):
        response = session.get(url, timeout=timeout)
        if response.status_code != 429 or attempt == 3:
            response.raise_for_status()
            return response
        retry_after = response.headers.get("Retry-After", "2")
        try:
            wait = min(10.0, max(1.0, float(retry_after)))
        except ValueError:
            wait = 2.0
        time.sleep(wait)
    raise RuntimeError(f"Request failed after retries: {url}")


def get_text(session: requests.Session, url: str, timeout: float) -> str:
    """Download a UTF-8 HTML page."""
    response = request(session, url, timeout)
    response.encoding = "utf-8"
    return response.text


def map_links(index_html: str, base_url: str) -> list[tuple[str, str]]:
    """Extract unique map detail links and names from an index page."""
    links = []
    seen = set()
    for match in re.finditer(r'<a\b([^>]*href=["\'][^"\']+["\'][^>]*)>([\s\S]*?)</a>', index_html, re.IGNORECASE):
        attrs = {key.lower(): unescape(value).strip() for key, _, value in ATTR_RE.findall(match.group(1))}
        href = attrs.get("href", "")
        parsed = urlparse(urljoin(base_url, href))
        map_match = MAP_LINK_RE.search(parsed.path)
        if not map_match or map_match.group(1) in seen:
            continue
        title = attrs.get("title", "") or re.sub(r"<[^>]+>", " ", match.group(2))
        title = " ".join(unescape(title).split())
        seen.add(map_match.group(1))
        links.append((map_match.group(1), urlunparse(parsed._replace(query="", fragment=""))))
    return links


def extract_attr(text: str, name: str) -> str:
    """Read one HTML attribute from a tag or element fragment."""
    match = re.search(rf'\b{re.escape(name)}\s*=\s*(["\'])(.*?)\1', text, re.IGNORECASE | re.DOTALL)
    return unescape(match.group(2)).strip() if match else ""


def parse_map(map_id: str, html: str, page_url: str) -> MapPage | None:
    """Parse a map detail page, returning None when it has no monsters."""
    section_match = MAP_SECTION_RE.search(html)
    if not section_match:
        return None
    section = section_match.group(1)
    image_tag = re.search(r"<img\b[^>]*data-map-image[^>]*>", section, re.IGNORECASE)
    image_url = urljoin(page_url, extract_attr(image_tag.group(0), "src")) if image_tag else ""
    stage = re.search(r"<(?:div|section)\b[^>]*data-map-stage[^>]*>", section, re.IGNORECASE)
    width = int(extract_attr(stage.group(0), "data-render-width") or 0) if stage else 0
    height = int(extract_attr(stage.group(0), "data-render-height") or 0) if stage else 0
    monsters = []
    for marker_match in MONSTER_RE.finditer(section):
        marker = marker_match.group(0)
        style = extract_attr(marker, "style")
        position = re.search(r"left\s*:\s*([\d.]+)%\s*;\s*top\s*:\s*([\d.]+)%", style, re.IGNORECASE)
        name = extract_attr(marker, "data-map-tooltip-name")
        marker_image = re.search(r"<img\b[^>]*>", marker, re.IGNORECASE)
        icon_url = urljoin(page_url, extract_attr(marker_image.group(0), "src")) if marker_image else ""
        if not position or not name or not icon_url:
            continue
        monsters.append(Monster(name=name, icon_url=icon_url, x=float(position.group(1)), y=float(position.group(2))))
    if not monsters or not image_url:
        return None
    title = extract_attr(section, "aria-label") or f"map-{map_id}"
    title = title.replace("地图全貌，可拖动并缩放", "").strip() or f"map-{map_id}"
    return MapPage(map_id, title, image_url, width, height, tuple(monsters))


def download_image(
    session: requests.Session, url: str, path: Path, timeout: float, width: int, height: int
) -> tuple[int, int]:
    """Download, scale, and flatten a map image onto the page background."""
    response = request(session, url, timeout)
    source = Image.open(io.BytesIO(response.content)).convert("RGBA")
    width = width or source.width
    height = height or source.height
    image = source.resize((width, height), Image.Resampling.LANCZOS)
    background = Image.new("RGBA", image.size, DEFAULT_BACKGROUND)
    background.alpha_composite(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    background.convert("RGB").save(path, format="PNG", optimize=True)
    return width, height


def compose_monsters(
    session: requests.Session,
    path: Path,
    monsters: tuple[Monster, ...],
    icon_cache: dict[str, Image.Image],
    timeout: float,
    width: int,
    height: int,
) -> None:
    """Composite the page's raw monster sprites at their marker coordinates."""
    image = Image.open(path).convert("RGBA")
    for monster in monsters:
        if monster.icon_url not in icon_cache:
            response = request(session, monster.icon_url, timeout)
            icon_cache[monster.icon_url] = Image.open(io.BytesIO(response.content)).convert("RGBA")
        icon = icon_cache[monster.icon_url].copy()
        icon.thumbnail((DEFAULT_MARKER_WIDTH, DEFAULT_MARKER_HEIGHT), Image.Resampling.LANCZOS)
        left = round(monster.x / 100 * width - DEFAULT_MARKER_WIDTH / 2 + (DEFAULT_MARKER_WIDTH - icon.width) / 2)
        top = round(monster.y / 100 * height - DEFAULT_MARKER_HEIGHT + (DEFAULT_MARKER_HEIGHT - icon.height) / 2)
        image.alpha_composite(icon, (left, top))
    image.convert("RGB").save(path, format="PNG", optimize=True)


def write_label(path: Path, monsters: tuple[Monster, ...], classes: dict[str, int], width: int, height: int) -> None:
    """Write marker boxes in normalized YOLO format."""
    marker_width = DEFAULT_MARKER_WIDTH / width
    marker_height = DEFAULT_MARKER_HEIGHT / height
    lines = []
    for monster in monsters:
        cx = monster.x / 100
        cy = (monster.y / 100) - marker_height / 2
        lines.append(f"{classes[monster.name]} {cx:.6f} {cy:.6f} {marker_width:.6f} {marker_height:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def page_count(index_html: str) -> int:
    """Return the pagination count, defaulting to one page."""
    match = PAGE_COUNT_RE.search(re.sub(r"<[^>]+>", "", index_html))
    return int(match.group(1)) if match else 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL, help="Map index URL")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--version", help="Site data version, for example CMSC or CMS079")
    parser.add_argument("--map-id", action="append", help="Export only this map ID; repeat for multiple IDs")
    parser.add_argument("--limit", type=int, help="Maximum number of map detail pages to inspect")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds")
    parser.add_argument("--timeout", type=float, default=30, help="Request timeout in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory")
    return parser.parse_args()


def main() -> None:
    """Download map images and labels."""
    args = parse_args()
    if args.delay < 0 or args.timeout <= 0:
        raise SystemExit("--delay must be non-negative and --timeout must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output is not empty: {output}. Use --overwrite to rebuild it.")
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Ultralytics map YOLO exporter/1.0"})
    index_url = set_version(args.index_url, args.version)
    index_html = get_text(session, index_url, args.timeout)
    if args.map_id:
        base = index_url.rsplit("/map/", 1)[0] + "/map/"
        links = [(map_id, urljoin(base, f"{map_id}/")) for map_id in dict.fromkeys(args.map_id)]
    else:
        links = []
        for page in range(1, page_count(index_html) + 1):
            html = (
                index_html
                if page == 1
                else get_text(session, set_version(f"{args.index_url}?page_num={page}", args.version), args.timeout)
            )
            links.extend(map_links(html, args.index_url))
            if page != page_count(index_html):
                time.sleep(args.delay)
        links = list(dict.fromkeys(links))
    if args.limit:
        links = links[: args.limit]
    if not links:
        raise SystemExit("No map detail pages found")

    classes: dict[str, int] = {}
    icon_cache: dict[str, Image.Image] = {}
    exported = 0
    skipped = 0
    for index, (map_id, link) in enumerate(links, start=1):
        detail_url = set_version(link, args.version)
        try:
            page = parse_map(map_id, get_text(session, detail_url, args.timeout), detail_url)
            if page is None:
                skipped += 1
                print(f"[{index}/{len(links)}] skip {map_id}: no monsters")
                continue
            for monster in page.monsters:
                classes.setdefault(monster.name, len(classes))
            image_path = output / "images" / f"map-{map_id}.png"
            label_path = output / "labels" / f"map-{map_id}.txt"
            width, height = download_image(session, page.image_url, image_path, args.timeout, page.width, page.height)
            compose_monsters(session, image_path, page.monsters, icon_cache, args.timeout, width, height)
            write_label(label_path, page.monsters, classes, width, height)
            exported += 1
            print(f"[{index}/{len(links)}] exported {map_id}: {len(page.monsters)} monster(s)")
        except (requests.RequestException, OSError, ValueError) as error:
            skipped += 1
            print(f"[{index}/{len(links)}] skip {map_id}: {error}", file=sys.stderr)
        if index != len(links):
            time.sleep(args.delay)

    (output / "classes.txt").write_text(
        "\n".join(name for name, _ in sorted(classes.items(), key=lambda item: item[1])) + "\n", encoding="utf-8"
    )
    dataset = {
        "path": output.as_posix(),
        "train": "images",
        "val": "images",
        "names": {class_id: name for name, class_id in classes.items()},
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Dataset created at {output}")
    print(f"Exported maps: {exported}; skipped maps: {skipped}; classes: {len(classes)}")


if __name__ == "__main__":
    main()
