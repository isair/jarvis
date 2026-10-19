# Structural Redaction

`redact` and `scrub_secrets` apply the same ordered, deterministic rules without
model inference or network access. They mask recognised email addresses, payment
card sequences, provider-shaped keys, authorisation headers, credential keyword
values, JWTs, long hexadecimal identifiers and contextual six-digit codes.

Email recognition starts at a local-part token boundary. Long ordinary tokens
and incomplete addresses are scanned without retrying every suffix; supported
email addresses retain their full redaction, including adjacent punctuation.

URI authority user information is removed before email matching, leaving a
credential-free URL. The rule applies to scheme names independently of the provider and
preserves the host, port, path, query and fragment for diagnosis. It supports
local hosts, IP addresses, IPv6 literals and percent-encoded credentials.
Repeated authority `@` characters are included in the masked user information.
Path and query `@` characters alone do not identify credentials.

`redact(text, max_len=8000)` collapses whitespace after scrubbing and limits the
result to the requested character count. `scrub_secrets(text)` preserves
whitespace, newlines and length for structured content and issue reports.

Redaction recognises structural patterns rather than proving arbitrary text is
free of secrets. Issue reporting keeps its local review and optional activity
log inclusion so the reporter can inspect the content before sharing.
