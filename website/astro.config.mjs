// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { SITE, GA_MEASUREMENT_ID } from './src/consts.ts';
import { rehypeDocLinks } from './src/rehype-doc-links.mjs';

// Inject Google Analytics into every Starlight page's <head>.
// (The custom landing page injects the same snippet via its own layout.)
const gaHead =
  GA_MEASUREMENT_ID && GA_MEASUREMENT_ID !== 'G-XXXXXXXXXX'
    ? [
        {
          tag: 'script',
          attrs: { async: true, src: `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}` },
        },
        {
          tag: 'script',
          content: `window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${GA_MEASUREMENT_ID}');`,
        },
      ]
    : [];

// https://astro.build/config
export default defineConfig({
  site: SITE.url,
  base: SITE.base,
  trailingSlash: 'ignore',
  markdown: {
    rehypePlugins: [[rehypeDocLinks, { base: SITE.base, docRoot: '../documentation' }]],
  },
  integrations: [
    starlight({
      title: SITE.title,
      description: SITE.description,
      tagline: SITE.tagline,
      favicon: '/favicon.svg',
      logo: {
        src: './src/assets/pipe.svg',
        alt: 'Language Pipes',
      },
      social: [
        { icon: 'github', label: 'GitHub', href: SITE.github },
      ],
      editLink: {
        baseUrl: `${SITE.github}/edit/main/documentation/`,
      },
      customCss: ['./src/styles/terminal.css', './src/styles/starlight.css'],
      head: [
        { tag: 'meta', attrs: { property: 'og:type', content: 'website' } },
        { tag: 'meta', attrs: { name: 'theme-color', content: '#0a0e0a' } },
        ...gaHead,
      ],
      components: {
        // Send the Starlight header's home link back to the custom landing page.
        SiteTitle: './src/components/SiteTitle.astro',
      },
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Install & Quick Start', link: '/#quick-start' },
            { label: 'Migrating to LP 2.0', slug: 'lp_2_migration' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'CLI Reference', slug: 'cli' },
            { label: 'Configuration', slug: 'configuration' },
            { label: 'OpenAI-Compatible API', slug: 'oai' },
          ],
        },
        {
          label: 'Architecture',
          items: [
            { label: 'Overview', slug: 'architecture' },
            { label: 'Job Processor State Machine', slug: 'job-processor' },
            { label: 'Request For Model Protocol', slug: 'request-for-model' },
            { label: 'Distributed State Network', slug: 'distributed-state-network' },
            { label: 'DSN Protocol', slug: 'distributed-state-network/protocol' },
            { label: 'DSN Node Config', slug: 'distributed-state-network/ds-node-config' },
            { label: 'DSN Node Server', slug: 'distributed-state-network/ds-node-server' },
            { label: 'LLM Layer Collector', slug: 'llm-layer-collector' },
          ],
        },
        {
          label: 'Privacy',
          items: [{ label: 'Threat Model', slug: 'privacy' }],
        },
        {
          label: 'Reference',
          items: [
            { label: 'Supported Models', slug: 'model_support' },
            { label: 'Release Notes', slug: 'release-notes' },
          ],
        },
      ],
    }),
  ],
});
