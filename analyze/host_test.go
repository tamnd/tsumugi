package analyze

import "testing"

func TestHostOf(t *testing.T) {
	cases := []struct{ in, want string }{
		{"https://www.Example.com/a/b?q=1", "www.example.com"},
		{"http://example.com:8080/x", "example.com"},
		{"https://shop.example.co.uk/", "shop.example.co.uk"},
		{"not a url", ""},
		{"ftp://example.com/x", ""},
	}
	for _, c := range cases {
		if got := HostOf(c.in); got != c.want {
			t.Errorf("HostOf(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestRegisteredDomain(t *testing.T) {
	cases := []struct{ in, want string }{
		{"www.example.com", "example.com"},
		{"a.b.example.com", "example.com"},
		{"shop.example.co.uk", "example.co.uk"},
		{"example.co.uk", "example.co.uk"},
		{"EXAMPLE.COM", "example.com"},
		{"example.com:443", "example.com"},
		{"", ""},
	}
	for _, c := range cases {
		if got := RegisteredDomain(c.in); got != c.want {
			t.Errorf("RegisteredDomain(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestRegisteredDomainCollapsesSubdomains is the property the distinct-linking-
// domain signal rests on: many subdomains a single owner controls collapse to one
// registered domain, so they count as one vote, not many.
func TestRegisteredDomainCollapsesSubdomains(t *testing.T) {
	a := RegisteredDomain("one.spam.example.com")
	b := RegisteredDomain("two.spam.example.com")
	c := RegisteredDomain("three.example.com")
	if a != b || a != c {
		t.Fatalf("subdomains did not collapse: %q %q %q", a, b, c)
	}
}
