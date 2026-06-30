package search

import "math"

// ForEachTerm calls fn for every distinct term in the broker's served shards with the term's
// document frequency summed across those shards, so a head node building the fleet-wide
// correction dictionary can pull one broker's whole vocabulary in one pass. Summing the df
// within the broker before handing it out keeps the wire carrying each term once rather than
// once per shard that holds it, and the head merges across brokers the same way (spell.Builder
// sums what it is given), so a term's df at the head is its true fleet-wide df.
//
// It is the broker-level counterpart to Shard.ForEachTerm: the pipeline builds a single-broker
// corrector by walking the shards directly, and a head node builds a fleet corrector by walking
// the brokers over the wire, both feeding the same builder. A concurrent publish or retire is
// handled the way every broker read is, against one loaded snapshot, so the enumeration is over
// a consistent shard set.
func (b *Broker) ForEachTerm(fn func(term string, df uint32)) {
	st := b.acquire()
	defer b.release(st)
	merged := make(map[string]uint64)
	for _, s := range st.shards {
		s.ForEachTerm(func(t string, df uint32) {
			merged[t] += uint64(df)
		})
	}
	for t, f := range merged {
		if f > math.MaxUint32 {
			f = math.MaxUint32
		}
		fn(t, uint32(f))
	}
}

// VectorDim reports the dense input dimension the broker's shards agree on and whether the
// dense plane is on at all, so a head node building the dense query encoder produces a vector
// of the width every shard beneath it can read. It mirrors the broker's own pipeline rule: the
// dense plane is on only when every shard that carries a vector region agrees on one dimension,
// and a fleet with no vector region, or one whose regions disagree, leaves it off rather than
// encode into a width some shard cannot read.
func (b *Broker) VectorDim() (int, bool) {
	st := b.acquire()
	defer b.release(st)
	dim, ok := 0, false
	for _, s := range st.shards {
		d, has := s.VectorDim()
		if !has {
			continue
		}
		if !ok {
			dim, ok = d, true
			continue
		}
		if d != dim {
			return 0, false
		}
	}
	return dim, ok
}
