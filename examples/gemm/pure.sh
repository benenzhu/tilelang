#!/bin/bash
set -euo pipefail

# Find all *gfx950.s files in /tmp
files=( *gfx950.s )

# Assert exactly one match
if [ ${#files[@]} -eq 0 ] || [ ! -e "${files[0]}" ]; then
    echo "ERROR: no *gfx950.s file found in /tmp" >&2
    exit 1
fi
if [ ${#files[@]} -ne 1 ]; then
    echo "ERROR: expected 1 gfx950.s file, found ${#files[@]}:" >&2
    printf '  %s\n' "${files[@]}" >&2
    exit 1
fi

f="${files[0]}"
echo "Found: $f"

# Generate pure.s (output: ${f}pure.s)
python /root/learn-hip/gen_pure.py "$f"

# Delete the original .s file
rm "$f"

# Open the cleaned assembly in cursor
cursor "${f}pure.s"
