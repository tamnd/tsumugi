package analyze

import (
	"net/url"
	"regexp"
	"strings"

	"github.com/tamnd/tsumugi/convert"
)

// mdInlineLinkText matches the same markdown inline links mdInlineLink does, but
// captures both the visible [text] (group 1) and the url (group 2), so the anchor
// extractor keeps the phrase Links throws away. The match positions are identical to
// mdInlineLink, so the targets it resolves are exactly the inline-link targets Links
// resolves; only the extra text group differs.
var mdInlineLinkText = regexp.MustCompile(`\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+"[^"]*")?\s*\)`)

// Anchor text extraction. A page's outbound links carry two things the offline
// build wants: the target, which Links already recovers into the link graph, and
// the visible text of the link, the phrase the linking author chose to describe
// the target. That phrase is inbound anchor text for the target, and the spec's
// doc 02 and doc 04 fold it into the target's anchor field, an off-page describes-me
// signal that often names a page in words the page never uses about itself.
//
// AnchorLink pairs the two so the collection build can invert them: gather every
// phrase pointing at a target across the corpus, and index them on the target. The
// target resolution here is identical to Links, down to the same normalization, so
// the anchor edges resolve to exactly the same documents the link graph does.

// AnchorLink is one outbound inline markdown link, carrying both its visible anchor
// text and its resolved absolute target URL. It is what AnchorLinks emits and what
// the anchor inversion keys on: the text becomes a phrase indexed on the document
// the URL resolves to.
type AnchorLink struct {
	Text string
	URL  string
}

// AnchorLinks returns the inline markdown links of one crawl document as (anchor
// text, normalized absolute target) pairs, in first-seen order. It mirrors Links:
// relative targets resolve against the page URL, targets normalize to the same
// canonical absolute form, and a link to the page's own URL is dropped. It differs
// in two ways. It keeps the visible [text] of each link, the phrase Links throws
// away, since that phrase is the whole point of an anchor. And it takes only inline
// links, not autolinks: an autolink <https://x> has no separate describing text, it
// is the bare URL, so it contributes a graph edge (via Links) but no anchor phrase.
//
// Links with empty or whitespace-only text are dropped, they name the target with
// nothing. The result is not deduplicated: a page that links a target twice under
// two phrases contributes both, and the inversion is what collapses repeats. The
// order is stable so a build over the same body yields the same phrases every run.
func AnchorLinks(d convert.Document) []AnchorLink {
	base, err := url.Parse(strings.TrimSpace(d.URL))
	if err != nil || base.Host == "" {
		return nil
	}
	self, _ := normalizeURL(base)

	var out []AnchorLink
	for _, m := range mdInlineLinkText.FindAllStringSubmatch(d.Body, -1) {
		text := normalizeAnchorText(m[1])
		if text == "" {
			continue
		}
		ref, err := url.Parse(strings.TrimSpace(m[2]))
		if err != nil {
			continue
		}
		abs, ok := normalizeURL(base.ResolveReference(ref))
		if !ok || abs == self {
			continue
		}
		out = append(out, AnchorLink{Text: text, URL: abs})
	}
	return out
}

// normalizeAnchorText reduces a raw markdown link text to the phrase the anchor
// field indexes: leading and trailing whitespace trimmed and every internal run of
// whitespace collapsed to one space, so "  see   the docs\n" and "see the docs" are
// one phrase. It caps the phrase at maxAnchorRunes so a pathological link with a
// whole paragraph as its text cannot blow up a target's anchor field; the cap is on
// runes, not bytes, so it never splits a multibyte character.
func normalizeAnchorText(raw string) string {
	text := strings.Join(strings.Fields(raw), " ")
	if r := []rune(text); len(r) > maxAnchorRunes {
		text = strings.TrimRight(string(r[:maxAnchorRunes]), " ")
	}
	return text
}

// maxAnchorRunes bounds a single anchor phrase. A normal link text is a few words;
// this leaves generous room for a descriptive title while capping the rare link
// whose text is an entire sentence or more so one source cannot flood a target.
const maxAnchorRunes = 120
