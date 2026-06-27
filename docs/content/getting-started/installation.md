---
title: "Installation"
description: "Install tsumugi from Go, Homebrew, Scoop, a release archive, a Linux package, or the container image."
weight: 20
---

tsumugi is a single static binary with no cgo and no runtime dependencies. Pick whichever channel suits you.

## Go

```bash
go install github.com/tamnd/tsumugi/cmd/tsumugi@latest
```

## Homebrew (macOS and Linux)

```bash
brew install tamnd/tap/tsumugi
```

## Scoop (Windows)

```bash
scoop bucket add tamnd https://github.com/tamnd/scoop-bucket
scoop install tsumugi
```

## Linux (apt and dnf)

A signed apt and dnf repository tracks every release, so `apt upgrade` and `dnf upgrade` keep tsumugi current.

```bash
# Debian, Ubuntu
curl -fsSL https://tamnd.github.io/linux-repo/gpg.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/tamnd.gpg
echo "deb [signed-by=/usr/share/keyrings/tamnd.gpg] https://tamnd.github.io/linux-repo/apt stable main" \
  | sudo tee /etc/apt/sources.list.d/tamnd.list
sudo apt update && sudo apt install tsumugi

# Fedora, RHEL
sudo dnf config-manager --add-repo https://tamnd.github.io/linux-repo/dnf/tamnd.repo
sudo dnf install tsumugi
```

## Release archives and Linux packages

Every [release](https://github.com/tamnd/tsumugi/releases) attaches `tar.gz` archives (and a `.zip` for Windows) for Linux, macOS, Windows, and FreeBSD, plus `.deb`, `.rpm`, and `.apk` packages and a `checksums.txt` with a cosign signature and SBOMs. Download the one for your platform, extract `tsumugi`, and put it on your `PATH`. To install a package directly without the repo above:

```bash
# Debian/Ubuntu
sudo dpkg -i tsumugi_*_amd64.deb

# Fedora/RHEL
sudo rpm -i tsumugi-*.x86_64.rpm
```

## Container

A multi-arch image ships on every release:

```bash
docker run --rm ghcr.io/tamnd/tsumugi version
```

Mount a collection in and serve it:

```bash
docker run --rm -p 8080:8080 -v "$PWD/data:/data" \
  ghcr.io/tamnd/tsumugi serve --dir /data --model /data/model.bin --addr :8080
```

## Verify it runs

```bash
tsumugi version
tsumugi --help
```

Next: [the quick start](/getting-started/quick-start/).
