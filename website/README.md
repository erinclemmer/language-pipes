# Language Pipes — Website

The documentation + landing site for [Language Pipes](https://github.com/erinclemmer/language-pipes),
built with [Astro](https://astro.build/) and [Starlight](https://starlight.astro.build/).

Live at **https://erinclemmer.github.io/language-pipes** (`languagepipes.com` redirects here).

## How it's wired

- **Landing page** — a fully custom, terminal-styled page at `src/pages/index.astro`.
- **Docs** — Starlight consumes the repo-root [`documentation/`](../documentation) folder
  **in place** at build time via a glob loader (`src/content.config.ts`). The
  `documentation/` folder stays the single source of truth; nothing is duplicated
  or moved, so every `documentation/…` link in the project README and code keeps
  working.
- Relative `.md` links inside the docs are rewritten to site URLs by a small rehype
  plugin (`src/rehype-doc-links.mjs`).
- Terminal theme lives in `src/styles/terminal.css` (landing) and
  `src/styles/starlight.css` (docs overrides).

## Develop

```bash
cd website
npm install
npm run dev        # http://localhost:4321/language-pipes/
```

| Command           | Action                                      |
| ----------------- | ------------------------------------------- |
| `npm run dev`     | Start the dev server                        |
| `npm run build`   | Build the production site to `dist/`        |
| `npm run preview` | Preview the production build locally        |

## Deploy

Pushing to `main` (touching `website/`, `documentation/`, or the workflow) triggers
`.github/workflows/deploy-website.yml`, which builds the site and publishes it to
GitHub Pages.

**One-time repo setup:** In **Settings → Pages**, set **Source** to
**GitHub Actions**.

## Google Analytics

Analytics is off by default. To enable it, set `GA_MEASUREMENT_ID` in
`src/consts.ts` to your GA4 property ID (format `G-XXXXXXXXXX`). The snippet is then
injected into both the landing page and every docs page.

## Custom domain

To serve `languagepipes.com` directly (instead of a redirect), add a `public/CNAME`
file containing the domain and update `site`/`base` in `astro.config.mjs`.
