// Command tsumugi builds, inspects, and serves .tsumugi shards: the compact
// single-file format for web-scale search and ranking.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/tamnd/tsumugi/cli"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	os.Exit(cli.Execute(ctx))
}
