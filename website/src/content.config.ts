import { defineCollection } from 'astro:content';
import { docsSchema } from '@astrojs/starlight/schema';
import { glob } from 'astro/loaders';

// The `documentation/` folder at the repo root remains the source of truth.
// We consume it at build time with a glob loader pointed one level up, so the
// docs never have to move and every README / AGENTS / test reference to
// `documentation/` keeps working.
export const collections = {
  docs: defineCollection({
    loader: glob({
      pattern: '**/[^_]*.{md,mdx}',
      base: '../documentation',
      // Map `dir/README.md` -> `dir` so the DSN README becomes a section index.
      generateId: ({ entry }) => {
        const slug = entry
          .replace(/\.(md|mdx)$/i, '')
          .replace(/(^|\/)README$/i, '$1')
          .replace(/\/$/, '');
        return slug || 'index';
      },
    }),
    schema: docsSchema(),
  }),
};
