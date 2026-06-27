package cli

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/tsumugi/collection"
	"github.com/tamnd/tsumugi/feature"
	"github.com/tamnd/tsumugi/search"
)

// TestServeCCrawlSchemaGuard builds a real multi-shard collection from the crawl export
// and proves the schema guard on the serve path: a model stamped with the canonical
// feature schema loads and serves, while a model stamped with a different schema is
// refused at openCollection rather than silently scoring on misaligned columns. This
// is the load-bearing guarantee at fleet scale, where shards and models are built over
// time and a schema drift would otherwise corrupt ranking without an error.
func TestServeCCrawlSchemaGuard(t *testing.T) {
	if _, err := os.Stat(ccrawlParquet); err != nil {
		t.Skipf("ccrawl parquet not present: %v", err)
	}
	if testing.Short() {
		t.Skip("skipping real-data build in short mode")
	}
	tmp := t.TempDir()
	out := filepath.Join(tmp, "coll")
	res, err := collection.Build(collection.Options{Source: ccrawlParquet, Out: out, ShardSize: 1000, Limit: 6000})
	if err != nil {
		t.Fatalf("Build from ccrawl: %v", err)
	}
	if res.Shards < 2 {
		t.Fatalf("need at least 2 shards, got %d", res.Shards)
	}

	// A model stamped with the canonical schema loads and serves over the real shards.
	good := filepath.Join(tmp, "good.bin")
	writeModelStamped(t, good, feature.SchemaVersion, feature.DefaultSchemaHash())
	broker, _, err := openCollection(out, good)
	if err != nil {
		t.Fatalf("canonical model refused on real collection: %v", err)
	}
	_ = broker.Close()

	// A model stamped with a different schema hash is refused before it can rank.
	bad := filepath.Join(tmp, "bad.bin")
	writeModelStamped(t, bad, feature.SchemaVersion, feature.DefaultSchemaHash()^0x1)
	if _, _, err := openCollection(out, bad); !errors.Is(err, search.ErrSchemaMismatch) {
		t.Fatalf("mismatched model got %v, want ErrSchemaMismatch", err)
	}

	// A model stamped with a different schema version is refused too.
	badVer := filepath.Join(tmp, "badver.bin")
	writeModelStamped(t, badVer, feature.SchemaVersion+1, feature.DefaultSchemaHash())
	if _, _, err := openCollection(out, badVer); !errors.Is(err, search.ErrSchemaMismatch) {
		t.Fatalf("mismatched version got %v, want ErrSchemaMismatch", err)
	}
}
