#!/usr/bin/env bash
# Download the SAAB SIRS 77 GHz FMCW radar dataset (Zenodo 10.5281/zenodo.5845259, CC BY 4.0)
# and verify the published md5 checksums. About 1.56 GB.
set -euo pipefail
DIR="${1:-data/raw}"
mkdir -p "$DIR"
BASE="https://zenodo.org/api/records/5845259/files"
NPY="data_SAAB_SIRS_77GHz_FMCW.npy"
NPY_MD5="01d66ba7b1ccc04477a9e69b2813f251"
README_MD5="767d670bec773ee7c2360fa9ae2fddf4"

fetch() {  # name md5
  local f="$DIR/$1"
  if [ -f "$f" ] && echo "$2  $f" | md5sum -c --status; then
    echo "ok (cached): $f"; return
  fi
  curl -L --fail --retry 5 -C - -o "$f" "$BASE/$1/content"
  echo "$2  $f" | md5sum -c -
}

fetch ReadMe.txt "$README_MD5"
fetch "$NPY" "$NPY_MD5"
