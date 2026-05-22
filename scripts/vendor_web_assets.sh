#!/usr/bin/env bash
# Refresh the vendored web UI assets in web/vendor/.
#
# Roitelet's "local-first" stance argues for the web client working
# offline. Self-hosting these scripts (~520 KB total) also removes a
# third-party supply-chain dependency. Re-run this script when you want
# to pull in a newer Tailwind or marked release; commit the result.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$(cd "${SCRIPT_DIR}/../web" && pwd)/vendor"
mkdir -p "${VENDOR_DIR}"

# (URL, output filename) pairs.
ASSETS=(
  "https://cdn.tailwindcss.com?plugins=typography|tailwindcss.js"
  "https://cdn.jsdelivr.net/npm/marked@12/marked.min.js|marked.min.js"
)

echo "Refreshing vendored web assets into ${VENDOR_DIR}"
for entry in "${ASSETS[@]}"; do
  url="${entry%%|*}"
  name="${entry##*|}"
  echo "  - ${name}  ←  ${url}"
  curl -fsSL --max-time 60 -o "${VENDOR_DIR}/${name}" "${url}"
done

echo
echo "Done. Sizes:"
ls -la "${VENDOR_DIR}"
