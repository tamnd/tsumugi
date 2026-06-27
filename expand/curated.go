package expand

// DefaultGroups is the small, deliberate curated expansion set a general web corpus
// ships with, the place a deployment encodes the exact, high-precision equivalences the
// learned doc-side plane does not reliably carry. It is intentionally short: the spec's
// rule is that query-side expansion stays light because the semantic heavy lifting
// happens offline, so this holds only acronyms and spelling or brand variants where
// exactness matters, not a broad thesaurus. A deployment extends it with its own domain
// groups and rebuilds the table.
//
// Each group lists raw forms; Build analyzes them, so casing and spacing here do not
// matter. Ambiguous short acronyms that collide with common words ("us", "la", "it")
// are deliberately left out, because a wrong expansion costs more than a missing one.
var DefaultGroups = []Group{
	// Place acronyms: the clearest acronym case, a single token expanding to a phrase.
	{"nyc", "new york city"},
	{"usa", "united states of america"},
	{"uk", "united kingdom"},
	{"eu", "european union"},
	// Technical acronyms common in a web corpus.
	{"ml", "machine learning"},
	{"ai", "artificial intelligence"},
	{"nlp", "natural language processing"},
	{"api", "application programming interface"},
	{"cpu", "central processing unit"},
	{"gpu", "graphics processing unit"},
	{"db", "database"},
	{"os", "operating system"},
	{"oss", "open source software"},
	// Spelling variants across English locales, the synonym case that is genuinely
	// bidirectional because both forms are single tokens.
	{"color", "colour"},
	{"gray", "grey"},
	{"center", "centre"},
	{"catalog", "catalogue"},
	{"organize", "organise"},
	{"analyze", "analyse"},
	{"license", "licence"},
	// Brand and abbreviation variants.
	{"js", "javascript"},
	{"ts", "typescript"},
	{"k8s", "kubernetes"},
	{"postgres", "postgresql"},
}

// Default builds the table over DefaultGroups with the given analysis chain, the table a
// broker constructs at startup when no deployment-specific groups are configured.
func Default(a Analyze) *Table {
	return Build(DefaultGroups, a)
}
