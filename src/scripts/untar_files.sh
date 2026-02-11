#!/bin/bash

# Variables
INPUT_DIR="/project_antwerp/dataset/downloads"
OUTPUT_DIR="/project_antwerp/dataset/decompressed"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all .tar.gz files in the input directory
for tar_file in "$INPUT_DIR"/*.tar.gz; do
    if [ -f "$tar_file" ]; then
        echo "Processing: $tar_file"
        # Extract the base name of the file (without extension)
        base_name=$(basename "$tar_file" .tar.gz)
        extract_dir="$OUTPUT_DIR/$base_name"

        # Check if extraction is needed
        if [ -d "$extract_dir" ] && [ "$(find "$extract_dir" -type f | wc -l)" -eq 10001 ]; then
            # Create a directory for the extracted files
            mkdir -p "$extract_dir"
            echo "Skipping extraction: $extract_dir already exists with 10001 files"
        else
            # Extract the tar.gz file
            tar -xzf "$tar_file" -C "$extract_dir" &
            echo "Started extracting: $tar_file to $extract_dir"
        fi
    fi
done

echo "All files started."
wait
echo "All files processed. Extracted files are located in: $OUTPUT_DIR"
