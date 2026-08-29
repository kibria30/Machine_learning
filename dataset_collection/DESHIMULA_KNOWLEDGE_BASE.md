# Deshimula Review Dataset — Knowledge Base

## What We Built

A scraper for **Deshimula** (https://deshimula.com), a Bangladeshi anonymous company review platform similar to Glassdoor, to create a structured dataset of employee reviews of Bangladeshi software/tech companies.

---

## 1. Site Overview

| Property | Value |
|---|---|
| Site | Deshimula (দেশি মূলা) |
| URL | https://deshimula.com |
| Purpose | Anonymous workplace stories for engineers, by engineers |
| Content | Company reviews (Positive, Negative, Mixed) |
| Language | Bengali + English mixed |
| Infrastructure | Cloudflare (has `cf_clearance` cookie) |
| Total companies | ~808 (directory page) |
| Total reviews | ~2,466 site-wide |

---

## 2. Technical Decisions

### Why `requests` over Selenium/Playwright?

Verified through live testing:
- The site is **server-side rendered** — reviews come baked into the initial HTML response.
- Zero XHR/fetch calls for review data in request logs — only ads, analytics, fonts.
- `SearchTerm`, `Vibe`, and pagination params all work server-side in the URL.
- Plain `requests` with proper `User-Agent` returns 200 OK — no Cloudflare challenge triggered during normal crawling.
- Only caveat: bot detection is possible at high request rates — handled with delay + retry/backoff.

---

## 3. URL Structure

### 3.1. Company Directory

```
GET /companies?page={1..27}
```

- 27 pages, ~30 companies per page = ~808 unique companies
- Each card shows: company slug, Deshi Mula review count, Glassdoor review count, Positive/Negative/Mixed breakdown, overall rating, recommend %

### 3.2. Company Profile

```
GET /companies/{slug}
```

- Exposes `companyId` in links to `/stories/1?companyId={id}`
- Exposes real company name in `<h1>` (directory uses obfuscated names)
- Some companies do NOT expose `companyId` (~30+ of 808) — requires SearchTerm fallback

### 3.3. Review Listing (Primary Method)

```
GET /stories/{page}?companyId={id}&Vibe={v}
```

| Param | Values |
|---|---|
| `companyId` | 24-char hex ID from company profile |
| `Vibe` | 1 = Positive, 2 = Negative, 3 = Mixed |
| `page` | 1-based pagination (30 reviews per page) |

**This is the reliable method** — proper pagination, exact company filtering.

### 3.4. Review Listing (SearchTerm Fallback)

```
GET /stories/{page}?SearchTerm={term}&Vibe={v}
```

- `SearchTerm` is the company name or slug
- Returns all matching reviews on page 1 (no pagination — page 2+ returns 0)
- **Fuzzy matching** — may return reviews from multiple companies → must filter by company slug link on card
- Used for companies that do not expose `companyId`

### 3.5. Review Detail

```
GET /story/{id}
```

- Returns full review text (listing page truncates behind "Read More")
- Content is in `<div class="story-content">`
- Does NOT show upvote/downvote/comment counts (those are only on listing cards)

---

## 4. Pagination & Scale

| Company | Positive | Negative | Mixed | Total |
|---|---|---|---|---|
| Brain Station 23 | 6 | 21 | 9 | 36 |
| Optimizely | 14 | 15 | 15 | 44 |

**Full site:**
- 723 companies have ≥1 review
- ~2,466 total reviews
- 2 companies with 51-100 reviews
- 19 with 21-50
- 30 with 11-20
- 672 with 1-10

**Important:** `SearchTerm` pagination is BROKEN — `/stories/2?SearchTerm=...` returns 0 reviews. Always use `companyId` for pagination.

---

## 5. HTML Parsing Strategy

### 5.1. Listing Card Structure

Each review on the listing page lives in a `div.container` block. The card contains:

```
div.container
├── div (header: company name + role + date)
│   ├── span.text-slate-950 → company nickname (obfuscated)
│   ├── div.mt-0.5 → role + date row
│   │   ├── span → "Software engineer"
│   │   ├── span → "•"
│   │   └── span → "May 31, 2025"
├── h2 → review title
├── div.text-slate-800 → review content (TRUNCATED)
└── div (footer: votes + comments)
    ├── Upvote SVG + span (number)
    ├── Downvote SVG + span (number)
    └── Comment SVG + span (number)
```

**Key observations:**
- Company name on cards is **obfuscated** (e.g., "Mogoj Station 23" instead of "Brain Station 23")
- Content is **truncated** — need detail page for full text
- Vote/comment counts: last 3 digit spans in the votes container (first numbers are badge counts from platform/Glassdoor)
- Story ID: from `<a href="/story/{id}">` link (24-char hex)

### 5.2. Detail Page Structure

```
div.story-content → full review text
h1 → review title
```

- No vote/comment counts on detail page
- Uses `.story-content` CSS class (not generic `text-slate-800`)

---

## 6. Scraper Architecture

### 6.1. Files

| File | Purpose |
|---|---|
| `deshimula_scraper.ipynb` | Original 2-company scraper (simple, no state) |
| `deshimula_all_companies.ipynb` | Full-site scraper with tiering, resumable state, rate limit |
| `deshimula_reviews.csv` | 2-company output (80 reviews) |
| `deshimula_all_reviews.csv` | Full-site output (all scraped reviews) |
| `deshimula_companies.csv` | Directory index (slug + review counts) |
| `state.json` | Resumable progress (done slugs + total IDs) |

### 6.2. Pipeline

```
1. Crawl /companies?page={1..27}
   → Collect slug + dm_review_count for each company
   → Skip companies with 0 reviews

2. Filter by tier (MIN_REVIEWS threshold)
   → Tier 1: ≥10 reviews (~59 companies, ~80% of data)
   → Tier 2: ≥1 reviews (~715 companies, full site)

3. For each company:
   a. GET /companies/{slug} → resolve companyId + real name
   b. If companyId found:
      → GET /stories/{page}?companyId={id}&Vibe={1|2|3}
      → Paginate until empty
   c. If companyId NOT found (fallback):
      → GET /stories/1?SearchTerm={slug}&Vibe={1|2|3}
      → Filter cards by company slug link
   d. For each review card:
      → Parse: title, role, date, upvote, downvote, comment_count
      → GET /story/{story_id} → full content
   e. Append to CSV, save progress to state.json
```

### 6.3. Rate Limiting & Retry

- **Delay:** 0.3 seconds between requests (configurable)
- **Retry:** 4 attempts with exponential backoff (2s, 5s, 10s, 20s + jitter)
- **Challenge detection:** Checks for `Just a moment` Cloudflare challenge page + HTTP 403/429
- **Threading:** Sequential (not parallel) — safer for avoiding blocks

### 6.4. Resumability

- Progress tracked in `state.json`: `{ done_slugs: [...], total_ids: N }`
- CSV appended incrementally (flushed per company)
- Re-running skips already-done companies
- Companies that failed are retried on next run

---

## 7. Dataset Schema

| Column | Type | Source | Notes |
|---|---|---|---|
| `id` | int | Auto-increment | Unique, sequential |
| `company_name` | string | Company profile `<h1>` | Real name, not obfuscated |
| `developer_role` | string | Listing card | e.g., "Software engineer", "Senior Developer" |
| `date` | string | Listing card | e.g., "May 31, 2025" |
| `title` | string | Listing card `<h2>` | Review headline |
| `content` | string | Detail page `.story-content` | Full review text (Bengali/English mixed) |
| `upvote` | int | Listing card | Number of upvotes |
| `downvote` | int | Listing card | Number of downvotes |
| `comment_count` | int | Listing card | Number of comments |
| `vibe` | string | URL param (1/2/3) | "Positive", "Negative", or "Mixed" |

---

## 8. Challenges & Solutions

| Challenge | Solution |
|---|---|
| Listing cards show obfuscated company names | Fetch real name from profile page `<h1>` |
| Listing content truncated ("Read More") | Fetch each `/story/{id}` detail page |
| `SearchTerm` pagination broken (page 2 = 0) | Use `companyId`-based pagination instead |
| Some companies have no `companyId` on profile | SearchTerm fallback with slug + card filtering |
| Cloudflare possible bot detection | Delay + User-Agent + retry/backoff |
| `__file__` undefined in Jupyter | Use `Path.cwd()` instead |
| Card vote counts include badge numbers | Take last 3 digit spans in votes container |
| Fuzzy SearchTerm matching returns wrong companies | Filter cards by company link href on each card |
| Duplicate reviews across vibes | Dedupe by (company_name, title, content) |

---

## 9. Request Estimates

| Scope | Companies | Reviews | Listing Pages | Detail Pages | Est. Time (0.3s) |
|---|---|---|---|---|---|
| 2 companies | 2 | 80 | 6 | 80 | ~30s |
| Tier 1 (≥10) | 59 | ~2,000 | ~120 | ~2,000 | ~10 min |
| Full site (≥1) | 715 | ~2,466 | ~800 | ~2,466 | ~25 min |

---

## 10. Legal & Ethics

### What the site says

**robots.txt content signals:**
```
Content-Signal: search=yes, ai-train=no, use=reference
```

- `search=yes` — building a search index is allowed
- `ai-train=no` — **training AI models is explicitly NOT permitted**
- `use=reference` — AI use limited to reference (short excerpts), not full text reproduction

**Terms & Conditions:**
- Reviews are user-generated content; opinions of users
- No usernames/PII displayed (anonymous platform)
- Governed by **Bangladesh law**, courts in Dhaka
- Contact: `deshimula@proton.me` (Cloudflare-obfuscated on Terms page, section 10)

### Risk Assessment

| Data Field | Risk Level | Reason |
|---|---|---|
| Company name | Low | Public factual data |
| Role, date | Low | Non-identifying, public |
| Vote/comment counts | Low | Public numerical data |
| Vibe label | Low | Derived from URL param |
| **Review text (content)** | **High** | Copyrighted UGC; `ai-train=no` applies |

### Recommendation

- **Private use only** (university submission) → low practical risk
- **Publishing** (Kaggle, GitHub, paper) → high risk without permission
- **Contact** `deshimula@proton.me` for permission before publication
- If publishing metadata only (no `content` column) → likely fine

---

## 11. How to Run

### 2-Company Scraper
```bash
cd dataset_collection
jupyter notebook deshimula_scraper.ipynb
# Run All → outputs deshimula_reviews.csv
```

### All-Company Scraper
```bash
cd dataset_collection
# Edit MIN_REVIEWS in deshimula_all_companies.ipynb (10 for Tier 1, 1 for full)
jupyter notebook deshimula_all_companies.ipynb
# Run All → outputs deshimula_all_reviews.csv + state.json
```

### Resume Interrupted Run
```bash
# Just re-run the notebook — it picks up from state.json
# Or delete state.json to start fresh
```

---

## 12. Extending the Dataset

### Adding more companies
Edit the `COMPANIES` list in `deshimula_scraper.ipynb`:
```python
COMPANIES = [
    ("brain-station-23", "Brain Station 23"),
    ("optimizely", "Optimizely"),
    ("new-company-slug", "New Company Name"),
]
```

### Changing tier threshold
Edit `MIN_REVIEWS` in `deshimula_all_companies.ipynb`:
- `MIN_REVIEWS = 10` → Tier 1 only (~59 companies)
- `MIN_REVIEWS = 1` → Full site (~715 companies)

### Adding new fields
Add to `FIELDS` list + `parse_card()` or `scrape_story()` as needed.
