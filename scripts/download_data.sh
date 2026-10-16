#!/usr/bin/env bash
# Download the FlyWire v783 connectome (as packaged by eonsystemspbc/fly-brain)
# and the FlyWire neuron annotations (Schlegel et al. 2024).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLY_BRAIN_COMMIT="a3db62f9436074e485c0278290c2164ed6150808"
FLY_BRAIN_RAW="https://raw.githubusercontent.com/eonsystemspbc/fly-brain/${FLY_BRAIN_COMMIT}/data"
ANNOT_URL="https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"

mkdir -p "$ROOT/data/raw/flywire_783" "$ROOT/data/raw/annotations"

fetch() {  # fetch <url> <dest>
    if [[ -s "$2" ]]; then
        echo "exists: ${2#$ROOT/}"
    else
        echo "downloading: ${2#$ROOT/}"
        curl -fSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
    fi
}

fetch "$FLY_BRAIN_RAW/2025_Connectivity_783.parquet" "$ROOT/data/raw/flywire_783/2025_Connectivity_783.parquet"
fetch "$FLY_BRAIN_RAW/2025_Completeness_783.csv"     "$ROOT/data/raw/flywire_783/2025_Completeness_783.csv"
fetch "$ANNOT_URL"                                   "$ROOT/data/raw/annotations/neuron_annotations.tsv"
