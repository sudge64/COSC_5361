#!/bin/bash

# Check if a directory argument was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_directory> [label_directory]"
    echo "Example: $0 /path/to/images /path/to/labels"
    exit 1
fi

# Check if the provided directory exists
if [ ! -d "$1" ]; then
    echo "Error: Directory '$1' does not exist."
    exit 1
fi

IMAGE_DIR="$1"
LABEL_DIR="${2:-}"  # Optional second argument for labels

# Count total images
shopt -s nullglob
images=("$IMAGE_DIR"/*.{jpg,jpeg,png,gif,bmp,tiff,tif,webp})
shopt -u nullglob

if [ ${#images[@]} -eq 0 ]; then
    echo "Error: No image files found in '$IMAGE_DIR'."
    exit 1
fi

echo "Found ${#images[@]} image(s) in '$IMAGE_DIR'"
if [ -n "$LABEL_DIR" ] && [ -d "$LABEL_DIR" ]; then
    echo "Label directory: '$LABEL_DIR'"
else
    echo "No label directory specified. Only images will be processed."
fi
echo

count=0

# Function to transform YOLO labels for rotation
rotate_labels() {
    local input_file="$1"
    local output_file="$2"
    local rotation="$3"  # 90, 180, or 270

    # Create empty output file
    > "$output_file"

    while IFS=' ' read -r class x_center y_center width height; do
        # Skip empty lines and comments
        [[ -z "$class" || "$class" =~ ^# ]] && continue

        local new_x new_y new_w new_h

        case "$rotation" in
            90)
                # 90° clockwise: x' = 1 - y, y' = x, w' = h, h' = w
                new_x=$(echo "1 - $y_center" | bc -l)
                new_y="$x_center"
                new_w="$height"
                new_h="$width"
                ;;
            180)
                # 180°: x' = 1 - x, y' = 1 - y
                new_x=$(echo "1 - $x_center" | bc -l)
                new_y=$(echo "1 - $y_center" | bc -l)
                new_w="$width"
                new_h="$height"
                ;;
            270)
                # 270° clockwise (90° counter-clockwise): x' = y, y' = 1 - x
                new_x="$y_center"
                new_y=$(echo "1 - $x_center" | bc -l)
                new_w="$height"
                new_h="$width"
                ;;
        esac

        # Format to 6 decimal places and write to output
        printf "%d %.6f %.6f %.6f %.6f\n" "$class" "$new_x" "$new_y" "$new_w" "$new_h" >> "$output_file"
    done < "$input_file"
}

for image in "${images[@]}"; do
    # Get filename without path and extension
    filename=$(basename "$image")
    name="${filename%.*}"
    ext="${filename##*.}"
    label_name="${name}.txt"

    echo "Processing: $filename"

    # Rotate 90 degrees
    convert "$image" -rotate 90 "${image%.*}_rotated90.${ext}"
    if [ $? -eq 0 ]; then
        echo "  ✓ Created: _rotated90.${ext}"

        # Handle labels for 90° rotation
        if [ -n "$LABEL_DIR" ] && [ -d "$LABEL_DIR" ] && [ -f "${LABEL_DIR}/${label_name}" ]; then
            rotate_labels "${LABEL_DIR}/${label_name}" "${LABEL_DIR}/${name}_rotated90.txt" 90
            echo "  ✓ Created: rotated label _rotated90.txt"
        fi
    else
        echo "  ✗ Failed to create: _rotated90.${ext}"
    fi

    # Rotate 180 degrees
    convert "$image" -rotate 180 "${image%.*}_rotated180.${ext}"
    if [ $? -eq 0 ]; then
        echo "  ✓ Created: _rotated180.${ext}"

        # Handle labels for 180° rotation
        if [ -n "$LABEL_DIR" ] && [ -d "$LABEL_DIR" ] && [ -f "${LABEL_DIR}/${label_name}" ]; then
            rotate_labels "${LABEL_DIR}/${label_name}" "${LABEL_DIR}/${name}_rotated180.txt" 180
            echo "  ✓ Created: rotated label _rotated180.txt"
        fi
    else
        echo "  ✗ Failed to create: _rotated180.${ext}"
    fi

    # Rotate 270 degrees
    convert "$image" -rotate 270 "${image%.*}_rotated270.${ext}"
    if [ $? -eq 0 ]; then
        echo "  ✓ Created: _rotated270.${ext}"

        # Handle labels for 270° rotation
        if [ -n "$LABEL_DIR" ] && [ -d "$LABEL_DIR" ] && [ -f "${LABEL_DIR}/${label_name}" ]; then
            rotate_labels "${LABEL_DIR}/${label_name}" "${LABEL_DIR}/${name}_rotated270.txt" 270
            echo "  ✓ Created: rotated label _rotated270.txt"
        fi
    else
        echo "  ✗ Failed to create: _rotated270.${ext}"
    fi

    echo
    ((count++))
done

echo "Done! Processed $count image(s)."
echo "Output files are saved in the same directory as the originals."
