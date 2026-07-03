# Language Pipes — Website Plan

A plan for a documentation + advertising site for Language Pipes. The goal is a
site that reads like a well-made open-source project homepage — terminal-native,
developer-focused — **not** a SaaS product page.

---

## 1. What we're selling

Language Pipes is a **peer-to-peer distributed inference engine for open-source
LLMs**. The site should lead with these differentiators, in priority order:

1. **Distributed inference** — splits a model's transformer layers across
   multiple machines, so a model too large for any single machine can run across
   a network of your own hardware.
2. **Privacy architecture** — only the node hosting the **End Model** ever sees
   raw text; every other node processes opaque hidden-state tensors. Backed by a
   real quantitative threat model (`documentation/privacy.md`), positioned
   against Petals. This is the credibility anchor — it is genuine research, not
   marketing.
3. **OpenAI-compatible API** — a drop-in `base_url` swap for existing OpenAI
   client code.
4. **Interactive TUI** — configuration, monitoring, and control, with an
   established ASCII-art / terminal identity.
5. **Open, Python-native** — `pip install language-pipes`, MIT-licensed, on
   GitHub and PyPI.

---

## 2. Technology

**Stack: [Astro](https://astro.build/) + [Starlight](https://starlight.astro.build/).**

Rationale:

- Ships static HTML with near-zero JavaScript — fast, cheap to host, ages well.
- Full design freedom for a **custom terminal-styled landing page**, while
  Starlight provides a batteries-included **docs** section (sidebar nav,
  built-in search, prev/next, syntax highlighting) out of the box.
- Consumes the existing `documentation/*.md` files with minimal changes —
  mostly adding frontmatter and adjusting relative links.
- Best fit for the "advertising **and** documentation" requirement: it's the
  only mainstream option that gives a fully custom homepage without fighting the
  framework, while keeping docs low-effort.

Alternatives considered and rejected:

| Option | Why not |
|--------|---------|
| MkDocs Material | Python-native and lowest-effort, but the landing page always reads as "a docs site," not a project homepage. Viable low-maintenance fallback. |
| Docusaurus | Powerful but React-heavy and tends to read corporate/SaaS — the exact vibe to avoid. |
| VitePress | Fast and clean, but landing-page theming is more constrained than Astro's. |

**Hosting / deployment:** GitHub Pages via a GitHub Actions workflow, alongside
the existing `.github/workflows/publish.yml`. Free, already on GitHub, no new
infrastructure. Custom domain can be added later via a `CNAME`.

**Repo layout:** a `website/` (or `site/`) directory in this repo keeps docs and
site versioned together. The Actions workflow builds `website/` and publishes to
Pages on push to `main`.

---

## 3. Visual direction

Terminal-native and dark-first. Lean into the identity the project already has
rather than inventing a new brand.

- **Type:** monospace throughout (or monospace headings + a clean sans body).
- **Palette:** dark "CRT" base — muted green/amber accents on near-black.
  Restrained, not neon.
- **Hero:** the existing ASCII-art logo, rendered as text (selectable), not an
  image.
- **Recurring motif:** the **pipe / network metaphor** — nodes passing tensors
  down a pipeline. Used for the "how it works" diagram and section dividers.
- **Explicitly avoid:** hero gradients, 3D product mockups, stock illustration,
  "Sign up free" energy. The reference point is a great `man` page crossed with
  a hacker project homepage.

---

## 4. Site structure

### Landing page (single scroll)

1. **Hero** — ASCII logo, tagline ("Peer-to-peer distributed inference for
   open-source language models"), `pip install language-pipes`, and buttons to
   Docs / GitHub / PyPI. Release + license + PyPI-downloads badges.
2. **The pitch** — run models too large for one machine, distributed across your
   own hardware, peer-to-peer.
3. **Three pillars** — *Distributed · Private · OpenAI-compatible*, each a short
   card linking to deeper docs.
4. **How it works** — the `TOKENIZE → EMBED → LAYER → NORM → HEAD` flow as a
   diagram, plus the layer-splitting explanation (layer models vs. end model).
5. **Privacy** — the End Model concept, teased with a prominent link to the full
   threat model. This is the section that separates the project from Petals.
6. **Quick start** — the condensed two-node example.
7. **Supported models + roadmap** — current families (Qwen3, Phi, Llama 3.1/3.2,
   Gemma 3) and planned work (INT8/INT4 quantization, GGUF, more architectures).
8. **Footer** — MIT license, PyPI, GitHub, and citation (from `CITATION.cff`).

### Docs section (Starlight sidebar)

Sourced from the existing `documentation/` tree:

- **Getting Started** — Installation, Quick Start, TUI tour.
- **Guides** — CLI Reference (`cli.md`), Configuration (`configuration.md`),
  OpenAI-Compatible API (`oai.md`).
- **Architecture** — Overview (`architecture.md`), Job Processor State Machine
  (`job-processor.md`), Distributed State Network
  (`distributed-state-network/`), LLM Layer Collector (`llm-layer-collector.md`).
- **Privacy** — full threat model (`privacy.md`).
- **Reference** — Supported models (`model_support.md`), LP 2 migration
  (`lp_2_migration.md`), Release notes (`release-notes.md`).

---

## 5. Build phases

1. **Scaffold** — Astro + Starlight project in `website/`, terminal theme
   (colors, monospace fonts, dark base), base layout and nav.
2. **Landing page** — build the 8 sections above with the custom theme.
3. **Docs migration** — move `documentation/*.md` in, add frontmatter, fix
   relative links, wire up the sidebar. Decide whether the repo docs become a
   pointer to the site or stay duplicated.
4. **Deploy** — GitHub Actions workflow → GitHub Pages; verify build on `main`.
5. **Polish** — the "how it works" pipeline diagram, TUI screenshots/recording,
   SEO/meta/OpenGraph, favicon.

---

## 6. Questions and Answers

- **Docs source of truth:** 
  Q: keep `documentation/*.md` as canonical and import at
  build time, or move them fully into `website/` and leave GitHub pointers?
  A: documentation folder is source of truth. If documentation needs to live under the website folder, prefer moving documentation to inside website folder and updating readme links
- **Domain:** 
  Q: GitHub Pages default (`erinclemmer.github.io/language-pipes`) or a
  custom domain?
  A: Build to work with github.io pages with google analytics. languagepipes.com will be redirected to erinclemmer.github.io/language-pipes
- **Diagram fidelity:** 
  Q: hand-built HTML/CSS pipeline diagram vs. a static image.
  A: prefer html/css