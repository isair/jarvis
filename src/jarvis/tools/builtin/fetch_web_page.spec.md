# Fetch web page tool

`fetchWebPage` fetches a supplied HTTP or HTTPS URL and returns extracted page text without an LLM call. Scheme-less URLs use HTTPS. Requests use a 15-second network timeout and follow redirects. Redirect bodies are closed before reading; Requests retains its redirect header and cookie handling.

- Responses are streamed in 64 KiB chunks. The retained decoded body is limited to 2 MiB, including decompressed gzip content and responses without a content length. An oversized body returns an unsuccessful tool result without partial page content. The response context closes on success, download errors and size rejection.
- BeautifulSoup removes script, style, metadata, stylesheet and noscript elements. Text extraction keeps the first 500 distinct non-empty lines longer than three characters, with an optional title.
- `include_links` enables up to 20 labelled links, resolving relative URLs against the supplied URL.
- Extracted replies are limited to 50,000 characters. Without BeautifulSoup, the bounded body is decoded using the response charset or Requests' character detector, with replacement for invalid bytes and UTF-8 fallback for unsupported encodings. Raw text is limited to 10,000 characters.
- Progress messages use emojis. Oversized pages produce an explicit download-limit message; debug logs identify the rejection without including page contents.
