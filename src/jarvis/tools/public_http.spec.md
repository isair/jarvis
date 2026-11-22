# Search page transport

`public_http.get` supplies streamed Requests responses for untrusted search-result
pages. It accepts HTTP and HTTPS URLs. Each request owns its Requests session
and closes session pools on return or failure. Origin authentication retains
Requests' behaviour, including host-specific netrc credentials. Callers close
streamed responses.
Redirect handling and cookies within a fetch retain Requests' normal behaviour.

Every new direct HTTP or HTTPS connection resolves its destination once. Empty
DNS answers, lookup failures and any non-global, reserved, site-local or multicast
address reject the connection before a socket is opened. IPv4-mapped IPv6 addresses use their IPv4
classification. Literal IP addresses are classified without DNS. A mixed public
and private answer is rejected as a whole.

Sockets connect only to a numeric address from the validated answer. The
connection hostname is retained for HTTP Host, TLS SNI and certificate checks.
No global DNS hooks or connection pool mappings are changed. Reused connections
remain attached to their validated socket; new connections validate afresh.
Configured proxies, proxy authentication, `NO_PROXY` and environment certificate
trust retain Requests' behaviour. Proxy requests validate the page hostname's
local DNS answers before sending, but the trusted configured proxy owns onward
DNS and routing. Numeric destination pinning applies only to direct connections.
Proxy endpoints may themselves be local. Requests owns origin and proxy
authentication handling, including its redirect credential rules.

Supplied-URL fetching uses Requests directly and retains local/intranet access,
proxy configuration and authentication behaviour.

Rejected connections produce debug logs without page content. Network failures
remain Requests exceptions so web tools return honest failures.
