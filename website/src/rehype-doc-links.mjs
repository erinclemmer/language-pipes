import path from 'node:path';

/**
 * Rewrites relative Markdown links (e.g. `./cli.md`,
 * `./distributed-state-network/README.md`) into the site's base-prefixed page
 * URLs. Needed because the `docs` collection is loaded from `../documentation`
 * (outside `src/`), where Astro's built-in relative-link resolution does not
 * fire. Absolute, external, and anchor-only links are left untouched.
 *
 * @param {{ base: string, docRoot: string }} options
 */
export function rehypeDocLinks({ base, docRoot }) {
  const trimmedBase = base.replace(/\/$/, '');
  const absDocRoot = path.resolve(docRoot);

  const toSlug = (rel) =>
    rel
      .replace(/\\/g, '/')
      .replace(/\.mdx?$/i, '')
      .replace(/(^|\/)README$/i, '$1')
      .replace(/(^|\/)index$/i, '$1')
      .replace(/\/$/, '');

  return (tree, file) => {
    const fileDir = file?.path ? path.dirname(file.path) : absDocRoot;

    const walk = (node) => {
      if (node.tagName === 'a' && node.properties && typeof node.properties.href === 'string') {
        const href = node.properties.href;
        const isRelativeMd =
          !/^(https?:|\/\/|\/|#|mailto:|tel:)/i.test(href) && /\.mdx?(#|$)/i.test(href);
        if (isRelativeMd) {
          const [rawPath, hash] = href.split('#');
          const abs = path.resolve(fileDir, rawPath);
          const slug = toSlug(path.relative(absDocRoot, abs));
          let url = slug ? `${trimmedBase}/${slug}/` : `${trimmedBase}/`;
          if (hash) url += `#${hash}`;
          node.properties.href = url;
        }
      }
      if (Array.isArray(node.children)) node.children.forEach(walk);
    };

    walk(tree);
  };
}
