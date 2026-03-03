#!/bin/bash

# Batch download script for SharePoint trajectory data
# Downloads files for replicas 2-8

# Configuration
OUTPUT_DIR="/project_antwerp/dataset/downloads_testing"
COOKIE_FEDAUTH="${FedAuth}"
COOKIE_RTFA="${rtFa}"
COMPOUND_NAME=$1
echo $COMPOUND_NAME
# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Base URL template
BASE_URL="https://imecinternational.sharepoint.com/:u:/r/sites/IDLab.Antwerp.Education.ThesisInternship-2025-202694-thesis-Sam.Roggeman/Shared%20Documents/thesis-Sam.Roggeman/Trajectory%20data/Cleaned%20pdb%20file%20tarballs"


# Download settings
TIMEOUT=300
MAX_RETRIES=3

echo "=========================================="
echo "SharePoint Batch Download Script"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Downloads: DPP 8-9 Replicas 2-8"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Function to download file with retries
download_file() {
    local dpp=$1
    local compound_name=$2
    local replica=$3
    local filename="sep_prot_frames_DPP${dpp}_${compound_name}_replica${replica}.tar.gz"
    local url="${BASE_URL}/${filename}"
    local output_file="${OUTPUT_DIR}/${filename}"
    local attempt=1
    
    echo "[DPP${dpp} Replica $replica] Downloading: $filename"
    
    # Check if file already exists
    if [ -f "$output_file" ] && [ $(stat -c%s "$output_file") -gt 1048576 ]; then
        echo "  ✓ Already exists ($(du -h "$output_file" | cut -f1))"
        return 0
    fi
    
    # Download with retries
    while [ $attempt -le $MAX_RETRIES ]; do
        echo -n "  Attempt $attempt/$MAX_RETRIES... "
        
        # Use wget with SharePoint-specific headers
        if wget \
            --timeout=$TIMEOUT \
            --tries=1 \
            --header="Cookie: FedAuth=$COOKIE_FEDAUTH; rtFa=$COOKIE_RTFA" \
            --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" \
            -q -O "$output_file.tmp" \
            "$url" 2>/dev/null; then
            
            # Check if download was successful (file size > 0)
        if [ $(stat -c%s "$output_file.tmp") -gt 1048576 ]; then
            mv "$output_file.tmp" "$output_file"
            local size=$(du -h "$output_file" | cut -f1)
            echo "✓ Downloaded ($size)"
            return 0
        else
            rm -f "$output_file.tmp"
            echo "✗ File too small (< 1MB)"
        fi
        else
            rm -f "$output_file.tmp"
            echo "✗ Failed"
        fi
        
        attempt=$((attempt + 1))
        if [ $attempt -le $MAX_RETRIES ]; then
            echo "    Waiting 5 seconds before retry..."
            sleep 5
        fi
    done
    
    echo "  ✗ Download failed after $MAX_RETRIES attempts"
    return 1
}

# Main download loop
TOTAL_FILES=7
SUCCESSFUL=0
FAILED=0
FAILED_REPLICAS=""
# Dpp 8 or 9
for dpp in 8 9; do
  for replica in {1..8}; do
      download_file $dpp $COMPOUND_NAME $replica
      if [ $? -eq 0 ]; then
          SUCCESSFUL=$((SUCCESSFUL + 1))
      else
          FAILED=$((FAILED + 1))
          FAILED_REPLICAS="$FAILED_REPLICAS $replica"
      fi
  done
done

# Summary
echo ""
echo "=========================================="
echo "Download Summary"
echo "=========================================="
echo "Total files: $TOTAL_FILES"
echo "Successful: $SUCCESSFUL"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo "Failed replicas:$FAILED_REPLICAS"
    echo ""
    echo "To retry failed downloads, run:"
    echo "  bash $0"
fi

echo ""
echo "Files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"/sep_prot_frames_DPP8_42_replica*.tar.gz 2>/dev/null | wc -l
echo "Total size:"
du -sh "$OUTPUT_DIR" 2>/dev/null

exit $FAILED