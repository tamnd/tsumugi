package lexical

import (
	"errors"
	"math"
	"sort"

	"github.com/tamnd/tsumugi/codec"
)

// The impact-ordered region is the doc-04 posting format. Where the BM25F region
// orders each list by docID and scores a document by the field-weighted term
// contribution, this one orders each list by descending impact and scores a document
// by the sum of the impacts of the query terms it carries. The impact is the
// document's composite static rank quantized to a byte, the same for every term the
// document contains, so with no learned per-term weights the score reduces to the
// query-term coverage weighted by static rank: a document that matches more of the
// query and ranks higher statically sorts first. The point of the ordering is early
// termination: because the impacts run high to low down each list and each block's
// leading impact is the largest it holds, a traversal that has filled its top-k can
// stop once the best any unseen document could reach falls below the k-th score, which
// the monotone block maxima make a cheap block-granularity test. This slice builds the
// format, the reader, and the exhaustive scorer that is the correctness oracle; the
// pruned traversal that exploits the ordering is the next slice, checked against this
// scorer.

// errNotImpactRegion is returned when an impact-scoring call is made against a region
// that was built docID-ordered. The two orderings decode their blocks differently, so
// scoring one as the other would misread the postings; the call is refused rather than
// returning garbage.
var errNotImpactRegion = errors.New("tsumugi/lexical: region is not impact-ordered")

// impactPosting is one entry in an impact-ordered list: a document and its one-byte
// impact. The impact is the document's quantized composite static rank, shared across
// every term the document contains, so it is stored once per posting rather than per
// field the way BM25F term frequencies are.
type impactPosting struct {
	docID  uint32
	impact uint8
}

// encodeImpactBlock writes one impact-ordered block. The block header is the block's
// minimum impact (its last posting's, since the list runs impact-descending) and the
// byte lengths of the two streams, the same three-varint shape the docID-ordered block
// uses so a reader steps to the next block from the header alone; only the first field's
// meaning changes, from a last-docID skip pointer to the min impact. The docIDs are
// zig-zag signed deltas against prevLast, the previous block's last docID, because
// impact order does not sort the docIDs so a gap can be negative. The payload is one
// impact byte per posting.
func encodeImpactBlock(out []byte, ps []impactPosting, prevLast uint32) []byte {
	minImpact := ps[len(ps)-1].impact

	var docs []byte
	prev := int64(prevLast)
	for _, p := range ps {
		docs = codec.AppendVarint(docs, int64(p.docID)-prev)
		prev = int64(p.docID)
	}

	pay := make([]byte, len(ps))
	for i := range ps {
		pay[i] = ps[i].impact
	}

	out = codec.AppendUvarint(out, uint64(minImpact))
	out = codec.AppendUvarint(out, uint64(len(docs)))
	out = codec.AppendUvarint(out, uint64(len(pay)))
	out = append(out, docs...)
	out = append(out, pay...)
	return out
}

// decodeImpactBlock decodes an impact block, given the last docID of the previous block
// so the first signed delta resolves to an absolute docID. The header field h.lastDocID
// carries the block's minimum impact here, not a docID, and is not needed for the decode
// (the payload holds every impact), so the docIDs come only from the signed gap stream.
// The payload length is the posting count, which the gap stream must match exactly.
func decodeImpactBlock(h blockHeader, prevLast uint32) ([]impactPosting, error) {
	ps := make([]impactPosting, 0, len(h.payload))
	prev := int64(prevLast)
	off := 0
	for off < len(h.docs) {
		d, n := codec.Varint(h.docs[off:])
		if n <= 0 {
			return nil, errCorrupt
		}
		off += n
		prev += d
		if prev < 0 || prev > math.MaxUint32 {
			return nil, errCorrupt
		}
		ps = append(ps, impactPosting{docID: uint32(prev)})
	}
	if len(ps) != len(h.payload) {
		return nil, errCorrupt
	}
	for i := range ps {
		ps[i].impact = h.payload[i]
	}
	return ps, nil
}

// BuildImpact encodes the accumulated documents into an impact-ordered lexical region.
// impactOf returns a document's one-byte impact, the quantized composite static rank the
// list ordering sorts on; the build calls it once per posting. It reuses the term set the
// docID-ordered Build accumulated, so the same corpus produces the same dictionary and the
// same bloom filter under either ordering, and only the posting bodies and their order
// differ.
func (b *Builder) BuildImpact(impactOf func(docID uint32) uint8) []byte {
	n := b.docCount()
	st := computeStats(n, b.norms)

	sorted := make([]string, 0, len(b.terms))
	for t := range b.terms {
		sorted = append(sorted, t)
	}
	sort.Strings(sorted)

	i := 0
	next := func() (string, []impactPosting, bool) {
		if i >= len(sorted) {
			return "", nil, false
		}
		term := sorted[i]
		i++
		docs := b.terms[term]
		ps := make([]impactPosting, 0, len(docs))
		for docID := range docs {
			ps = append(ps, impactPosting{docID: docID, impact: impactOf(docID)})
		}
		// Impact-descending, docID-ascending on a tie, so the order is deterministic and
		// the block maxima are the leading impacts.
		sort.Slice(ps, func(a, c int) bool {
			if ps[a].impact != ps[c].impact {
				return ps[a].impact > ps[c].impact
			}
			return ps[a].docID < ps[c].docID
		})
		return term, ps, true
	}

	return assembleImpactRegion(n, st, b.params, b.codec, next)
}

// impactTermSource yields terms in ascending dictionary order, each with its postings
// already sorted impact-descending. It is the impact-ordered counterpart of termSource;
// the in-memory Builder drives assembleImpactRegion through it, and an external-merge
// build could too.
type impactTermSource func() (term string, ps []impactPosting, ok bool)

// assembleImpactRegion encodes an impact-ordered region from a sorted term stream. It
// mirrors assembleRegion's layout so the region opens through the same header and the same
// bloom, dictionary, block-max, and postings sections: only the posting encoding and the
// meaning of the block-max array change. The block-max array holds each block's leading
// (largest) impact, and maxContrib is the whole list's largest impact, the bounds the
// pruned traversal reads. There is no norms section, because impact scoring needs no field
// lengths. dc names the docID gap codec in the header for format symmetry, but impact
// blocks decode their signed deltas directly, so its id is recorded and not otherwise used.
func assembleImpactRegion(n uint32, st stats, params Params, dc docCodec, next impactTermSource) []byte {
	var postings []byte
	var blockMax []byte
	var sorted []string
	var entries []termEntry

	for {
		term, ps, ok := next()
		if !ok {
			break
		}
		ti := len(sorted)
		df := uint32(len(ps))

		entry := termEntry{
			termID:      uint32(ti),
			postingsOff: uint64(len(postings)),
			docFreq:     df,
			blockMaxOff: uint64(len(blockMax)),
		}

		var prevLast uint32
		var listMax int32
		var blockCount uint32
		for start := 0; start < len(ps); start += blockSize {
			end := start + blockSize
			if end > len(ps) {
				end = len(ps)
			}
			block := ps[start:end]
			// The list runs impact-descending, so the block's leading posting carries its
			// largest impact. That is the block-max the pruning reads, and because the whole
			// list is sorted the leading impacts are monotone non-increasing across blocks.
			bmax := int32(block[0].impact)
			if bmax > listMax {
				listMax = bmax
			}
			blockMax = codec.AppendUint32(blockMax, uint32(bmax))
			postings = encodeImpactBlock(postings, block, prevLast)
			prevLast = block[len(block)-1].docID
			blockCount++
		}

		entry.postingsLen = uint64(len(postings)) - entry.postingsOff
		entry.blockCount = blockCount
		entry.maxContrib = listMax

		sorted = append(sorted, term)
		entries = append(entries, entry)
	}

	dictBytes := encodeDict(sorted, entries)

	bf := newBloom(len(sorted), 0.01)
	for _, t := range sorted {
		bf.add(t)
	}
	bloomBytes := bf.encode()

	h := regionHeader{
		flags:       flagIDFFreeBlockMax | flagImpactMode,
		docidCodec:  dc.id(),
		termCount:   uint32(len(sorted)),
		docCount:    n,
		avgFieldLen: st.avgFieldLen,
		params:      params,
	}
	off := uint64(headerLen)
	h.bloomOff = off
	h.bloomLen = uint64(len(bloomBytes))
	off += h.bloomLen
	h.dictOff = off
	h.dictLen = uint64(len(dictBytes))
	off += h.dictLen
	h.blockMaxOff = off
	h.blockMaxLen = uint64(len(blockMax))
	off += h.blockMaxLen
	h.postingsOff = off
	h.postingsLen = uint64(len(postings))
	off += h.postingsLen
	h.normsOff = off
	h.normsLen = 0

	out := h.encode()
	out = append(out, bloomBytes...)
	out = append(out, dictBytes...)
	out = append(out, blockMax...)
	out = append(out, postings...)
	return out
}

// eachImpactPosting decodes every posting of an impact-ordered list and calls fn for each,
// threading the last docID from one block into the next so the signed deltas resolve to
// absolute docIDs.
func (r *Region) eachImpactPosting(e termEntry, fn func(impactPosting)) error {
	list := r.postings[e.postingsOff : e.postingsOff+e.postingsLen]
	off := 0
	var prevLast uint32
	for i := uint32(0); i < e.blockCount; i++ {
		h, err := readBlockHeader(list, off)
		if err != nil {
			return err
		}
		ps, err := decodeImpactBlock(h, prevLast)
		if err != nil {
			return err
		}
		for _, p := range ps {
			fn(p)
		}
		if len(ps) > 0 {
			prevLast = ps[len(ps)-1].docID
		}
		off = h.nextOffset
	}
	return nil
}

// exhaustiveImpact scores every document that contains any query term by summing the
// impacts of the terms it carries, with no pruning, and keeps the top-k. It decodes every
// block of every list, so it is the correctness oracle the pruned impact traversal is
// checked against, the impact-ordered counterpart of exhaustive.
func (r *Region) exhaustiveImpact(infos []termInfo, k int) ([]Candidate, error) {
	if !r.impact {
		return nil, errNotImpactRegion
	}
	acc := map[uint32]int32{}
	for _, info := range infos {
		err := r.eachImpactPosting(info.entry, func(p impactPosting) {
			acc[p.docID] += int32(p.impact)
		})
		if err != nil {
			return nil, err
		}
	}
	tk := newTopK(k)
	for docID, score := range acc {
		tk.offer(Candidate{DocID: docID, Score: score})
	}
	return tk.results(), nil
}

// SearchImpact returns the top-k documents for a query over an impact-ordered region,
// scoring by query-term coverage weighted by static rank. It serves from the pruned
// traversal, which decodes the lists highest impact first and stops once the top-k is
// settled; the exhaustive scan it is gated against is exhaustiveImpact.
func (r *Region) SearchImpact(query string, k int) ([]Candidate, error) {
	return r.SearchImpactTerms(Analyze(query), k)
}

// SearchImpactTerms is SearchImpact over an already-analyzed term set, the analyze-once
// path the broker takes.
func (r *Region) SearchImpactTerms(terms []string, k int) ([]Candidate, error) {
	if !r.impact {
		return nil, errNotImpactRegion
	}
	infos := r.termInfos(terms, nil)
	if len(infos) == 0 {
		return nil, nil
	}
	return r.prunedImpact(infos, k)
}

// impactBlockInvariant decodes every block of every impact-ordered list and checks the
// ordering properties the pruning rests on: the postings within a block run impact-
// descending, the stored block-max is the block's leading impact, the header's min-impact
// is the block's last impact, the leading impacts are monotone non-increasing across
// blocks, and maxContrib is the first block's leading impact. It is exposed for the test
// that asserts the format directly rather than only through the oracle.
func (r *Region) impactBlockInvariant() (bool, error) {
	if !r.impact {
		return false, errNotImpactRegion
	}
	for ti := uint32(0); ti < r.terms; ti++ {
		term, ok := r.dict.term(ti)
		if !ok {
			return false, errCorrupt
		}
		e, ok := r.lookupEntry(term)
		if !ok {
			return false, errCorrupt
		}
		list := r.postings[e.postingsOff : e.postingsOff+e.postingsLen]
		off := 0
		var prevLast uint32
		var prevLeading int32 = math.MaxInt32
		for bi := uint32(0); bi < e.blockCount; bi++ {
			h, err := readBlockHeader(list, off)
			if err != nil {
				return false, err
			}
			ps, err := decodeImpactBlock(h, prevLast)
			if err != nil {
				return false, err
			}
			if len(ps) == 0 {
				return false, nil
			}
			leading := int32(ps[0].impact)
			stored := int32(codec.Uint32(r.blockMax[e.blockMaxOff+uint64(bi)*4:]))
			if stored != leading {
				return false, nil
			}
			if leading > prevLeading {
				return false, nil
			}
			if int32(h.lastDocID) != int32(ps[len(ps)-1].impact) {
				return false, nil
			}
			for i := 1; i < len(ps); i++ {
				if ps[i].impact > ps[i-1].impact {
					return false, nil
				}
			}
			if bi == 0 && e.maxContrib != leading {
				return false, nil
			}
			prevLeading = leading
			prevLast = ps[len(ps)-1].docID
			off = h.nextOffset
		}
	}
	return true, nil
}
