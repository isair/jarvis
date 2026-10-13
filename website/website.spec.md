# Public product website

The public Jarvis website describes the desktop assistant, its setup and its privacy and licensing boundaries. It is separate from the assistant runtime and imports no user configuration or memory code.

## Content and identity

- Product identity, repository, canonical base and page metadata live in `site.json`.
- Product, getting-started and privacy pages contain complete, readable HTML without client-side rendering.
- Claims describe supported capabilities and their practical limitations. Personal/non-commercial use is free; commercial use requires a separate licence. The source is publicly available under the custom licence.
- Local processing requires local models/endpoints. Downloads, web features and connected tools have their own network requirements.
- Screenshots use curated public assets from `docs/img`. No user stores, attachments or runtime data enter the website.
- The README stays concise. Public website links supplement its existing installation and feature information.

## Discovery and access

- Each page has a unique title/description, one canonical URL and social preview metadata. The XML sitemap lists the canonical pages.
- SoftwareApplication metadata represents the visible product description and licence. It contains no invented ratings, reviews or performance guarantees.
- The configured GitHub Pages base URL controls publication, including an inherited account domain or an individual custom domain. Project-path links work under that base.
- Standard useful, crawlable content supports web and AI discovery. No search rank or AI citation is guaranteed; there are no hidden instructions for answer engines.
- Navigation is keyboard-accessible, the skip link focuses main without another tab stop, images have descriptive alternatives, layouts support narrow screens and motion honours the reduced-motion preference.
- No analytics, external scripts, external fonts or forms collect visitor data.

## Build and publication

- The standard-library builder renders a static bundle without installing the assistant or loading models.
- Public canonical URLs use HTTPS without credentials, query strings, fragments or relative path segments.
- Output directories containing unrelated files or symbolic links are rejected. A rebuild recognises its generated manifest and refuses added or missing files.
- Pull requests validate the site without deployment credentials. Public deployment is restricted to trusted develop changes and uses minimum GitHub Pages/OIDC permissions.
- Public deployment is independent of stable desktop releases and does not change the repository's default branch.
