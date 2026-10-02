---
title: "Source: README.md"
type: source-page
doc: README.md
updated: 2026-07-03
---
# README.md — repo build instructions

Repo-maintenance doc for the Jekyll site itself: how to build/serve locally, and a
`bin/convert_to_single_md.py` script that flattens all chapters into one
`scaling-book-combined.md` (stripping Jekyll frontmatter, converting figure includes to plain
markdown images, and normalizing math/LaTeX) for conversion to a Word document via pandoc. No
TPU-performance content; purely a contributor-facing build/tooling doc.

## Feeds
- No topic page — this doc is build tooling, not book content, and is represented by this source
  page alone (coverage floor).
