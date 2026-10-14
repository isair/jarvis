# Fetch web page tool

`fetchWebPage` fetches a supplied HTTP or HTTPS URL and returns extracted page text without an LLM call. Scheme-less URLs use HTTPS. Requests follow redirects with a 15-second network timeout. The shared `tools/http_response.py` hook closes redirect bodies before reading while retaining Requests' redirect header and cookie handling.

- Decoded responses stream in 64 KiB chunks using urllib3 2.8 or later for bounded decompression. The read loop checks its 15-second deadline between chunks and reads at most 32 MiB of decoded content. Redirect and network waits retain the Requests timeout contract.
- The `url` field must be a JSON string and `include_links`, when supplied, must be a JSON boolean. Invalid fields fail before a request. The omitted link flag defaults to false.
- A local lxml HTML target parser emits text events without constructing a page DOM or retaining the complete response. Scripts, styles, noscript and template content are excluded. No page scripts, subresources or external entities are fetched by parsing.
- A bounded initial prefix retains encoding metadata across small transport chunks. A byte-order mark, explicit HTTP charset or initial HTML declaration selects the encoding; missing or unsupported encodings use UTF-8. Unicode decoding continues across chunk boundaries.
- `PageText` retains at most 48,000 text characters, a 512-character title and 20 links with bounded labels and URLs. Text extraction returns at most 500 distinct non-empty lines longer than three characters. Relative links resolve against the final response URL.
- Replies contain at most 50,000 characters including metadata and any excerpt label. Reaching a download, reading or content limit returns available readable text with an explicit partial-page label. A bounded read that obtains no readable text fails honestly. Empty pages do not provide successful evidence.
- Responses close on success, limits and exceptions. Progress messages use emojis. Debug logs record decoded-byte and retained-text counts and whether the result is partial, without page contents.
