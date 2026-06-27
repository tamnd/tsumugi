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
