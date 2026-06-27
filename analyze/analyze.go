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
