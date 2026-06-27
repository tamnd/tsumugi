package spell

// Builder accumulates the collection-wide correction dictionary by merging per-shard
// term enumerations, summing each term's per-shard document frequency into the
// fleet-wide df the rank tie-break uses. The corrector must be collection-wide, not
// per-shard, because a per-shard corrector would suggest different corrections
// depending on which shard happened to hold the term, so the broker feeds every
// shard's ForEachTerm into Add and then calls Build once.
//
// Keeping the merge here, over a plain (term, df) callback, keeps the spell package
// free of a dependency on the index packages: the broker bridges its shards to the
// builder, the builder owns the merge.
type Builder struct {
	freqs map[string]uint64
}

// NewBuilder returns an empty collection-wide correction-dictionary builder.
func NewBuilder() *Builder {
	return &Builder{freqs: make(map[string]uint64)}
}

// Add folds one shard's term and its local document frequency into the merge, summing
// the df across shards so a term that appears in several shards ends with its
// fleet-wide df. It is shaped to be passed straight to Region.ForEachTerm.
func (b *Builder) Add(term string, df uint32) {
	if term == "" {
		return
	}
	b.freqs[term] += uint64(df)
}

// Len reports the number of distinct terms accumulated so far.
func (b *Builder) Len() int { return len(b.freqs) }

// Build finalizes the merged dictionary into a SymSpell index at the given maximum
// edit distance with no length or frequency cutoffs. The accumulated terms are already
// unique, so the index's per-term deletes are enumerated once over the fleet-wide
// vocabulary.
func (b *Builder) Build(maxEdit int) *Index {
	return b.BuildWithOptions(Options{MaxEdit: maxEdit})
}

// BuildWithOptions finalizes the merged dictionary honoring the length and frequency
// cutoffs, the variant a real collection uses so the long tail of crawl noise does not
// blow up the delete index.
func (b *Builder) BuildWithOptions(o Options) *Index {
	terms := make([]TermFreq, 0, len(b.freqs))
	for t, f := range b.freqs {
		terms = append(terms, TermFreq{Term: t, Freq: f})
	}
	return BuildWithOptions(terms, o)
}
