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
        # Create a directory for the extracted files
        extract_dir="$OUTPUT_DIR/$base_name"
        mkdir -p "$extract_dir"
        # Extract the tar.gz file
        tar -xzf "$tar_file" -C "$extract_dir" &
        echo "Extracted" "$tar_file" to "$extract_dir"
    fi
done

echo "All files started."
wait
echo "All files processed. Extracted files are located in: $OUTPUT_DIR"
