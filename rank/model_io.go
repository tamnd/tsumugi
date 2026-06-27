package rank

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
	"math"
)

// modelMagic tags a serialized ensemble file.
var modelMagic = [4]byte{'T', 'R', 'N', 'K'}

// modelVersion is the current TRNK format. Version 2 added the feature-matrix
// schema stamp (version and hash) after the tree count; version 1 carried no stamp
// and is still read, its schema left zero so the loader treats it as unstamped.
const modelVersion = 2

// ErrBadModel is returned when a model stream does not parse.
var ErrBadModel = errors.New("rank: bad model stream")

// Save writes the ensemble as a portable model file: a magic and version, the
// feature width and tree count, then each tree in preorder. The trained ensemble
// is the durable artifact a serving node loads, so this is the boundary between the
// offline trainer and the online evaluator.
func (e *Ensemble) Save(w io.Writer) error {
	bw := bufio.NewWriter(w)
	if _, err := bw.Write(modelMagic[:]); err != nil {
		return err
	}
	var hdr [12]byte
	hdr[0] = modelVersion
	binary.LittleEndian.PutUint32(hdr[4:], uint32(e.numFeatures))
	binary.LittleEndian.PutUint32(hdr[8:], uint32(len(e.trees)))
	if _, err := bw.Write(hdr[:]); err != nil {
		return err
	}
	// The schema stamp follows the v1 header: the feature-matrix schema version and
	// its fingerprint, so a serving node can refuse a model whose columns do not line
	// up with the shards it would score.
	var schema [12]byte
	binary.LittleEndian.PutUint16(schema[0:], e.schemaVersion)
	binary.LittleEndian.PutUint64(schema[4:], e.schemaHash)
	if _, err := bw.Write(schema[:]); err != nil {
		return err
	}
	for _, t := range e.trees {
		if err := writeNode(bw, t); err != nil {
			return err
		}
	}
	return bw.Flush()
}

func writeNode(w io.Writer, n *treeNode) error {
	if n.leaf {
		if _, err := w.Write([]byte{0}); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, n.value)
	}
	if _, err := w.Write([]byte{1}); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, int32(n.feature)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, n.thresh); err != nil {
		return err
	}
	if err := writeNode(w, n.left); err != nil {
		return err
	}
	return writeNode(w, n.right)
}

// LoadEnsemble reads an ensemble written by Save.
func LoadEnsemble(r io.Reader) (*Ensemble, error) {
	br := bufio.NewReader(r)
	var magic [4]byte
	if _, err := io.ReadFull(br, magic[:]); err != nil {
		return nil, err
	}
	if magic != modelMagic {
		return nil, ErrBadModel
	}
	var hdr [12]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil {
		return nil, err
	}
	if hdr[0] != 1 && hdr[0] != modelVersion {
		return nil, ErrBadModel
	}
	numFeatures := int(binary.LittleEndian.Uint32(hdr[4:]))
	numTrees := int(binary.LittleEndian.Uint32(hdr[8:]))
	// Version 2 carries the schema stamp after the header; version 1 carries none, so
	// its schema stays zero and the loader treats the model as unstamped.
	var schemaVersion uint16
	var schemaHash uint64
	if hdr[0] >= 2 {
		var schema [12]byte
		if _, err := io.ReadFull(br, schema[:]); err != nil {
			return nil, err
		}
		schemaVersion = binary.LittleEndian.Uint16(schema[0:])
		schemaHash = binary.LittleEndian.Uint64(schema[4:])
	}
	trees := make([]*treeNode, numTrees)
	for i := range trees {
		t, err := readNode(br)
		if err != nil {
			return nil, err
		}
		trees[i] = t
	}
	return &Ensemble{
		trees:         trees,
		numFeatures:   numFeatures,
		schemaVersion: schemaVersion,
		schemaHash:    schemaHash,
	}, nil
}

func readNode(r io.ByteReader) (*treeNode, error) {
	tag, err := r.ReadByte()
	if err != nil {
		return nil, err
	}
	switch tag {
	case 0:
		v, err := readFloat(r)
		if err != nil {
			return nil, err
		}
		return newLeaf(v), nil
	case 1:
		feat, err := readInt32(r)
		if err != nil {
			return nil, err
		}
		thresh, err := readFloat(r)
		if err != nil {
			return nil, err
		}
		left, err := readNode(r)
		if err != nil {
			return nil, err
		}
		right, err := readNode(r)
		if err != nil {
			return nil, err
		}
		return newSplit(int(feat), thresh, left, right), nil
	default:
		return nil, ErrBadModel
	}
}

func readFloat(r io.ByteReader) (float64, error) {
	u, err := readUint64(r)
	return math.Float64frombits(u), err
}

func readInt32(r io.ByteReader) (int32, error) {
	var u uint32
	for i := 0; i < 4; i++ {
		b, err := r.ReadByte()
		if err != nil {
			return 0, err
		}
		u |= uint32(b) << (8 * i)
	}
	return int32(u), nil
}

func readUint64(r io.ByteReader) (uint64, error) {
	var u uint64
	for i := 0; i < 8; i++ {
		b, err := r.ReadByte()
		if err != nil {
			return 0, err
		}
		u |= uint64(b) << (8 * i)
	}
	return u, nil
}
