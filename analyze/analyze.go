// Package analyze derives a document's title and its query-independent features from
// the crawled text and url. It is the bridge between the raw crawl document and the
// feature matrix the ranking model reads: everything that can be computed from one
// document in isolation is computed here, in one place, so the build pipeline stays a
// thin loop over a source.
//
// The features here are the content and url signals, the ones a single page yields on
// its own. The link-graph signals, PageRank and the in-degree family, need the whole
// graph and are filled by the graph stage when a crawl carries outlinks; a markdown
// export without links leaves them at zero, which a model trained on the same schema
// reads as the absence of those signals rather than a wrong value.
package analyze

import (
	"net/url"
	"strings"
	"unicode"

	"github.com/tamnd/tsumugi/convert"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/lexical"
)

// Analysis is the derived view of a document: a title for display and the feature
// values keyed by id for the feature matrix.
type Analysis struct {
	Title    string
	Features map[feature.FeatureID]float64
}

// maxTitle bounds a derived title so a heading-less page with a long first line does
// not store a paragraph as its title.
const maxTitle = 160

// Document derives the title and the content and url features of one crawl document.
func Document(d convert.Document) Analysis {
	title := deriveTitle(d.Body)
	bodyTokens := len(lexical.Analyze(d.Body))
	titleTokens := len(lexical.Analyze(title))

	depth, ulen, https, urlTokens := urlFeatures(d.URL)
	quality := contentQuality(d.Body)
	latin := latinRatio(d.Body)
	boiler := boilerplateRatio(d.Body)

	// A simple static rank: reward content, penalize depth, the prior every web
	// ranker starts from before the link graph refines it.
	static := float64(bodyTokens)
	if static > 4000 {
		static = 4000
	}
	static = static/40 - float64(depth)*3
	if static < 0 {
		static = 0
	}

	feats := map[feature.FeatureID]float64{
		feature.FeatStaticRank:     static,
		feature.FeatDocLen:         float64(bodyTokens),
		feature.FeatBodyLen:        float64(bodyTokens),
		feature.FeatTitleLen:       float64(titleTokens),
		feature.FeatURLDepth:       float64(depth),
		feature.FeatURLLen:         float64(ulen),
		feature.FeatURLFieldLen:    float64(urlTokens),
		feature.FeatHTTPS:          boolFeat(https),
		feature.FeatContentQuality: quality,
		feature.FeatBoilerplate:    boiler,
		feature.FeatLanguage:       latin,
	}
	return Analysis{Title: title, Features: feats}
}

// deriveTitle takes the first markdown heading, or the first non-empty line if the
// page has no heading, trimmed of markdown punctuation and bounded in length.
func deriveTitle(body string) string {
	for _, raw := range strings.Split(body, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			line = strings.TrimSpace(strings.TrimLeft(line, "#"))
		}
		line = strings.Trim(line, "*_`[]<>|")
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if len(line) > maxTitle {
			line = strings.TrimSpace(line[:maxTitle])
		}
		return line
	}
	return ""
}

// urlFeatures returns the path depth, the url length, whether it is https, and the
// number of tokens in the url, the url-field signals.
func urlFeatures(raw string) (depth, length int, https bool, tokens int) {
	length = len(raw)
	u, err := url.Parse(raw)
	if err != nil {
		return 0, length, false, len(lexical.Analyze(raw))
	}
	https = u.Scheme == "https"
	for _, seg := range strings.Split(u.Path, "/") {
		if seg != "" {
			depth++
		}
	}
	tokens = len(lexical.Analyze(u.Host + " " + u.Path))
	return depth, length, https, tokens
}

// contentQuality is a cheap text-density heuristic: the fraction of the body that is
// letters or digits rather than markup and whitespace, scaled to a 0..100 range. A
// boilerplate-heavy page scores low, a prose page high.
func contentQuality(body string) float64 {
	if body == "" {
		return 0
	}
	var alnum int
	for _, r := range body {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			alnum++
		}
	}
	return float64(alnum) / float64(len(body)) * 100
}

// boilerplateLinkDensity is the share of a block's visible text that must sit inside
// link anchors for the block to count as boilerplate. At half or more of the text
// being link labels the block is a nav bar, a footer link row, or a link list rather
// than prose, the markdown shape of page chrome.
const boilerplateLinkDensity = 0.5

// boilerplateRatio estimates the fraction of a page's visible text that is chrome
// (navigation, link rows, footers, link lists) rather than main content. The spec's
// extractor walks the parsed blocks and scores each by text density, classifying the
// low-density high-link blocks as boilerplate; this corpus stores the crawl as
// markdown, which keeps both the block structure (blank lines separate blocks) and
// the link markup, so the markdown analog scores each block by its link density. A
// block whose visible text is mostly link labels is chrome; a block of prose carries
// few link characters relative to its text. The ratio is the visible-text length of
// the boilerplate blocks over the total visible-text length.
//
// A high ratio (most of the page is link chrome) is a negative quality signal:
// doorway pages, auto-generated tag indexes, and thin pages are mostly link lists
// around a sliver of content. A low ratio means the page is mostly substance. The
// split is the same content/chrome separation a readability mode produces, and the
// content side is what would feed snippets and term extraction, so the ratio is a
// byproduct of work the build does anyway.
func boilerplateRatio(body string) float64 {
	var boiler, total int
	for _, block := range textBlocks(body) {
		visible, anchor := blockDensity(block)
		if visible == 0 {
			continue
		}
		total += visible
		if float64(anchor)/float64(visible) >= boilerplateLinkDensity {
			boiler += visible
		}
	}
	if total == 0 {
		return 0
	}
	return float64(boiler) / float64(total)
}

// textBlocks splits a markdown body into blocks, the runs of consecutive non-blank
// lines separated by blank lines, the unit the boilerplate extractor scores. It is
// the markdown stand-in for the parsed DOM blocks the spec's extractor walks.
func textBlocks(body string) []string {
	var blocks []string
	var cur []string
	for _, line := range strings.Split(body, "\n") {
		if strings.TrimSpace(line) == "" {
			if len(cur) > 0 {
				blocks = append(blocks, strings.Join(cur, "\n"))
				cur = nil
			}
			continue
		}
		cur = append(cur, line)
	}
	if len(cur) > 0 {
		blocks = append(blocks, strings.Join(cur, "\n"))
	}
	return blocks
}

// blockDensity returns a block's visible-text length and how much of it lies inside
// link anchor labels, both measured as letters and digits so markup punctuation,
// list markers, emphasis, and whitespace count toward neither. A markdown link is
// [label](target): the label is visible text and counts as anchor, the target is
// markup and counts as nothing, which is what makes a row of links read as
// all-anchor (boilerplate) and a paragraph with one citation read as mostly prose.
func blockDensity(block string) (visible, anchor int) {
	rs := []rune(block)
	for i := 0; i < len(rs); {
		if rs[i] == '[' {
			j := i + 1
			for j < len(rs) && rs[j] != ']' {
				j++
			}
			if j < len(rs) && j+1 < len(rs) && rs[j+1] == '(' {
				n := alnumCount(rs[i+1 : j])
				visible += n
				anchor += n
				k := j + 2
				for k < len(rs) && rs[k] != ')' {
					k++
				}
				if k < len(rs) {
					k++ // consume the closing paren of the target
				}
				i = k
				continue
			}
		}
		if unicode.IsLetter(rs[i]) || unicode.IsDigit(rs[i]) {
			visible++
		}
		i++
	}
	return visible, anchor
}

// alnumCount counts the letters and digits in a rune slice, the visible-text measure
// blockDensity uses on both link labels and prose so the two are comparable.
func alnumCount(rs []rune) int {
	var n int
	for _, r := range rs {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			n++
		}
	}
	return n
}

// latinRatio is one when the body is mostly latin-script letters and zero when it is
// mostly another script, a stand-in language signal until a real detector lands.
func latinRatio(body string) float64 {
	var letters, latin int
	for _, r := range body {
		if !unicode.IsLetter(r) {
			continue
		}
		letters++
		if r < 0x250 {
			latin++
		}
	}
	if letters == 0 {
		return 0
	}
	if float64(latin)/float64(letters) > 0.5 {
		return 1
	}
	return 0
}

func boolFeat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
