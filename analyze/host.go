package analyze

import (
	"strings"

	"golang.org/x/net/publicsuffix"
)

// HostOf returns the lowercased host of a URL with any port stripped, the grouping
// key the host-rank aggregation buckets pages into. It returns the empty string
// when the URL has no usable host.
func HostOf(raw string) string {
	cu, ok := CanonicalURL(raw)
	if !ok {
		return ""
	}
	// CanonicalURL already lowercased the host and dropped a default port; a
	// non-default port stays, so strip it here for the grouping key.
	host := cu[strings.Index(cu, "://")+3:]
	if i := strings.IndexByte(host, '/'); i >= 0 {
		host = host[:i]
	}
	if i := strings.LastIndexByte(host, ':'); i >= 0 {
		host = host[:i]
	}
	return host
}

// RegisteredDomain returns the registered domain (eTLD+1) of a host, the grouping
// key the domain-rank aggregation and the distinct-linking-domain count bucket
// pages into. www.example.co.uk and shop.example.co.uk both reduce to
// example.co.uk, so a farm cannot manufacture independent-looking domains under one
// it controls. It falls back to the host itself when the public suffix list cannot
// reduce it (a bare host, an IP, an unknown suffix).
func RegisteredDomain(host string) string {
	host = strings.ToLower(strings.TrimSpace(host))
	if host == "" {
		return ""
	}
	if i := strings.LastIndexByte(host, ':'); i >= 0 {
		host = host[:i]
	}
	if d, err := publicsuffix.EffectiveTLDPlusOne(host); err == nil {
		return d
	}
	return host
}
