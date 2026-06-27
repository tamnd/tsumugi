package vector

// dotI8 computes the integer dot product of an int8 query against an int8
// document vector, accumulating in int32 to avoid overflow. This is the portable
// rerank kernel; an avo-generated AVX2 path is a later optimization, noted in the
// impl note, and the pure-Go loop the compiler vectorizes modestly is the first
// cut.
func dotI8(q, v []int8) int32 {
	var s int32
	for i := range v {
		s += int32(q[i]) * int32(v[i])
	}
	return s
}

// dotF32I8 computes the asymmetric dot of a full-precision query against an int8
// document vector, accumulating in float64. It is the rerank kernel the spec calls
// for (05 line 544): the document stays int8 (it is what the region stores), but the
// query keeps its full precision instead of being quantized, so the query's
// quantization error never enters the score. Multiply the result by the int8 scale to
// land in the dequantized dot space; for a cosine the scale cancels against the
// document norm. The portable loop is the first cut; an AVX2 path is a later kernel.
func dotF32I8(q []float32, v []int8) float64 {
	var s float64
	for i := range v {
		s += float64(q[i]) * float64(v[i])
	}
	return s
}
