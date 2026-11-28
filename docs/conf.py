"""Sphinx configuration for the cbls documentation site (#466).

Build it from the repository root (docs/development.md, Documentation site):

    uv run --only-group docs sphinx-build -W -b html docs build/docs/html

The C++ reference needs `doxygen`; this file runs it on the supported headers
before Sphinx reads the pages. The ctest test docs_build names the binary in
CBLS_DOXYGEN and its output directory, in the build directory, in
CBLS_DOXYGEN_OUT.
"""

import html
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx

ROOT = Path(__file__).resolve().parent.parent


def _describe() -> str:
    """The version the site is built from, for the sidebar and llms.txt.

    A tag deploy names its tag (v0.2.0); a build from main names the tag it
    follows and its commit: main (after v0.1.0, abc1234), or main (abc1234)
    before the first tag. The docs workflow checks out the full history, without
    which git knows no tags. With no git, CMakeLists.txt's project(VERSION).
    """
    try:
        described = subprocess.run(
            ["git", "describe", "--tags", "--match", "v[0-9]*", "--always"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        described = ""
    if after := re.fullmatch(r"(.+)-\d+-g([0-9a-f]+)", described):
        return f"main (after {after[1]}, {after[2]})"
    if re.fullmatch(r"[0-9a-f]+", described):  # no tag in the history
        return f"main ({described})"
    if described:
        return described
    cmake = (ROOT / "CMakeLists.txt").read_text()
    found = re.search(r"project\(cbls VERSION ([0-9.]+)", cmake)
    return f"v{found[1]}" if found else "unknown version"


# The one configure found, unless it has gone since: then PATH's.
DOXYGEN = os.environ.get("CBLS_DOXYGEN", "")
if not os.path.isfile(DOXYGEN):
    DOXYGEN = shutil.which("doxygen") or ""
DOXYGEN_OUT = Path(os.environ.get("CBLS_DOXYGEN_OUT") or ROOT / "build" / "docs" / "doxygen")

# The supported API (README, Use from C++; #447): <cbls/cbls.h>, the headers it
# includes, and <cbls/io_nl.h>. Read from cbls.h, so a header it gains is
# documented too; the push paths in .github/workflows/docs.yml take
# include/cbls/** for the same reason.
_CBLS_H = ROOT / "include" / "cbls" / "cbls.h"
CPP_HEADERS = [
    "include/cbls/cbls.h",
    *(f"include/cbls/{h}" for h in re.findall(r'^#include "(\w+\.h)"', _CBLS_H.read_text(), re.M)),
    "include/cbls/io_nl.h",
]

DOXYGEN_CONFIG = f"""
PROJECT_NAME     = cbls
INPUT            = {" ".join(f'"{ROOT / h}"' for h in CPP_HEADERS)}
STRIP_FROM_PATH  = "{ROOT / "include"}"
OUTPUT_DIRECTORY = "{DOXYGEN_OUT}"
EXTRACT_ALL      = YES
GENERATE_XML     = YES
GENERATE_HTML    = NO
GENERATE_LATEX   = NO
QUIET            = YES
WARN_IF_UNDOCUMENTED = NO
WARN_IF_DOC_ERROR = NO
"""
# WARN_IF_DOC_ERROR: the headers' comments are written for their readers, not
# for doxygen; `O(#nodes + #rows)` there means counts, not links.

if not DOXYGEN:
    raise RuntimeError("the docs build needs doxygen on PATH for the C++ reference")
DOXYGEN_OUT.mkdir(parents=True, exist_ok=True)
subprocess.run([DOXYGEN, "-"], input=DOXYGEN_CONFIG, text=True, check=True, cwd=ROOT)

project = "cbls"
author = "Simon Spoorendonk"
copyright = "2026, Simon Spoorendonk"
release = _describe()
# The short form: the release's number, or that of the tag main follows.
_number = re.search(r"\d+\.\d+\.\d+", release)
version = _number[0] if _number else release

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "breathe",
    "autoapi.extension",
    "sphinx_llms_txt",
    "sphinxext.opengraph",
]

# The site's pages are the README (index.md), the results, and the references.
# The other files under docs/ are for working on cbls and are read on GitHub;
# the pages link them by absolute URL.
exclude_patterns = [
    "architecture.md",
    "benchmarks/**",
    "development.md",
    "git-hooks.md",
    "inner-solver-hooks.md",
    "measuring.md",
    "profiling.md",
]

html_theme = "furo"
# Furo shows it in the sidebar, under the logo.
html_title = f"cbls {release}"
html_baseurl = "https://spoorendonk.dk/cbls/"

# The site's identity: an accent colour (Furo's brand variables; no custom
# CSS), and a logo and favicon cut from the landing page's figure. An amber,
# apart from cptp's teal; the light one keeps link contrast on white.
# figures/make_routes.py draws the figure and the logos in these colours, and
# the favicon in an amber between them; change them there too.
ACCENT_LIGHT = "#b45309"
ACCENT_DARK = "#f59e0b"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": ACCENT_LIGHT,
        "color-brand-content": ACCENT_LIGHT,
    },
    "dark_css_variables": {
        "color-brand-primary": ACCENT_DARK,
        "color-brand-content": ACCENT_DARK,
    },
    "light_logo": "logo-light.svg",
    "dark_logo": "logo-dark.svg",
}
# The logos and the link-preview card are named one by one, so that _static
# gets no copy of the script.
html_static_path = [
    "figures/logo-light.svg",
    "figures/logo-dark.svg",
    "figures/routes-card.png",
]
html_favicon = "figures/favicon.svg"

# The site in a sentence: the landing page's description in link previews and
# search results, and llms.txt's summary.
SUMMARY = (
    "cbls is a constraint-based local search engine for mixed discrete-continuous "
    "models with non-convex, transcendental or discontinuous terms: ViolationLS "
    "over an expression DAG, as a C++17 library, a command-line solver and a "
    "Python module."
)

# Open Graph tags, so that a link to the site shows a card on LinkedIn and
# elsewhere. The image is a PNG, as they render no SVG; figures/make_routes.py
# draws it, and its size is read from its PNG header.
CARD = ROOT / "docs" / "figures" / "routes-card.png"
CARD_WIDTH, CARD_HEIGHT = struct.unpack(">II", CARD.read_bytes()[16:24])
ogp_site_url = html_baseurl
ogp_site_name = "cbls"
ogp_image = "_static/routes-card.png"
ogp_image_alt = "Routes cbls finds for a CVRP instance, fanning out from a central depot."
ogp_custom_meta_tags = (
    f'<meta property="og:image:width" content="{CARD_WIDTH}" />',
    f'<meta property="og:image:height" content="{CARD_HEIGHT}" />',
    '<meta name="twitter:card" content="summary_large_image" />',
)
# No generated cards: they need matplotlib, which the docs group does not bring.
ogp_social_cards = {"enable": False}


def _describe_landing_page(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: Any
) -> None:
    """Give the root page SUMMARY as its description, and the site's URL.

    Runs before sphinxext-opengraph's handler, which reads the page's metadata
    for og: overrides and adds no description meta tag where one exists. The
    canonical link (`pageurl`) names the root too, as og:url does.
    """
    if pagename != app.config.root_doc:
        return
    context["pageurl"] = html_baseurl
    context["meta"] = {
        **(context.get("meta") or {}),
        "og:description": SUMMARY,
        "og:url": html_baseurl,
    }
    context["metatags"] += f'<meta name="description" content="{html.escape(SUMMARY)}" />\n'


def setup(app: Sphinx) -> None:
    app.connect("html-page-context", _describe_landing_page, priority=400)


myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3

breathe_projects = {"cbls": str(DOXYGEN_OUT / "xml")}
breathe_default_project = "cbls"
breathe_default_members = ("members", "undoc-members")

# Read statically from the committed stub and sources: the docs build does
# not compile the extension.
autoapi_dirs = [str(ROOT / "python" / "cbls")]
autoapi_file_patterns = ["*.pyi", "*.py"]
autoapi_add_toctree_entry = False

llms_txt_summary = f"{SUMMARY} This is the documentation of cbls {release}."
# llms-full.txt concatenates the page sources, where the README's MyST
# include and the Breathe directives stay unexpanded: append what they render.
llms_txt_code_files = [f"+:../{f}" for f in ["README.md", *CPP_HEADERS]]
# For the same reason llms.txt links the rendered pages, not their sources.
llms_txt_uri_template = "{base_url}{docname}.html"
