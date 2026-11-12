# Public web page transport

`public_http.get` supplies streamed Requests responses for supplied pages and
search-result pages. It accepts HTTP and HTTPS URLs without embedded credentials.
Each request owns its Requests session, ignores ambient proxy and netrc settings,
and closes session pools on return or failure. Callers close streamed responses.
Redirect handling and cookies within a fetch retain Requests' normal behaviour.

Every new HTTP or HTTPS connection resolves its destination once. Empty DNS
answers, lookup failures and any non-global, reserved, site-local or multicast address reject the
connection before a socket is opened. IPv4-mapped IPv6 addresses use their IPv4
classification. Literal IP addresses are classified without DNS. A mixed public
and private answer is rejected as a whole.

Sockets connect only to a numeric address from the validated answer. The
connection hostname is retained for HTTP Host, TLS SNI and certificate checks.
No global DNS hooks or connection pool mappings are changed. Reused connections
remain attached to their validated socket; new connections validate afresh.
Rejected connections produce debug logs without page content. Network failures
remain Requests exceptions so web tools return honest failures.
