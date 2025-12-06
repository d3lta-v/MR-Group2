#!/bin/bash

# Script: copy_images.sh
# Description: Copy images from paths listed in a text file to a destination directory
# Usage: ./copy_images.sh <input_file> <destination_directory> [options]

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PRESERVE_STRUCTURE=false
VERBOSE=false
DRY_RUN=false

# Function to print usage
usage() {
    echo "Usage: $0 <input_file> <destination_directory> [options]"
    echo ""
    echo "Arguments:"
    echo "  input_file            Text file containing image paths (one per line)"
    echo "  destination_directory Target directory to copy images to"
    echo ""
    echo "Options:"
    echo "  -p, --preserve        Preserve directory structure"
    echo "  -v, --verbose         Verbose output"
    echo "  -d, --dry-run         Show what would be copied without actually copying"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 image_list.txt ./output_images"
    echo "  $0 image_list.txt ./output_images --preserve --verbose"
    exit 1
}

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
if [ $# -lt 2 ]; then
    print_error "Insufficient arguments"
    usage
fi

INPUT_FILE="$1"
DEST_DIR="$2"
shift 2

# Parse options
while [ $# -gt 0 ]; do
    case "$1" in
        -p|--preserve)
            PRESERVE_STRUCTURE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate input file
if [ ! -f "$INPUT_FILE" ]; then
    print_error "Input file does not exist: $INPUT_FILE"
    exit 1
fi

# Create destination directory if it doesn't exist
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$DEST_DIR"
    if [ $? -ne 0 ]; then
        print_error "Failed to create destination directory: $DEST_DIR"
        exit 1
    fi
fi

# Initialize counters
TOTAL_FILES=0
SUCCESS_COUNT=0
SKIP_COUNT=0
ERROR_COUNT=0

print_info "Starting image copy operation..."
print_info "Input file: $INPUT_FILE"
print_info "Destination: $DEST_DIR"
print_info "Preserve structure: $PRESERVE_STRUCTURE"
[ "$DRY_RUN" = true ] && print_warning "DRY RUN MODE - No files will be copied"
echo ""

# Read file line by line
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    
    # Trim whitespace
    line=$(echo "$line" | xargs)
    
    ((TOTAL_FILES++))
    
    # Check if source file exists
    if [ ! -f "$line" ]; then
        print_warning "File not found: $line"
        ((ERROR_COUNT++))
        continue
    fi
    
    # Determine destination path
    if [ "$PRESERVE_STRUCTURE" = true ]; then
        # Preserve directory structure
        RELATIVE_DIR=$(dirname "$line")
        TARGET_DIR="$DEST_DIR/$RELATIVE_DIR"
        TARGET_FILE="$TARGET_DIR/$(basename "$line")"
        
        if [ "$DRY_RUN" = false ]; then
            mkdir -p "$TARGET_DIR"
        fi
    else
        # Copy to flat directory
        TARGET_FILE="$DEST_DIR/$(basename "$line")"
    fi
    
    # Check if file already exists at destination
    if [ -f "$TARGET_FILE" ] && [ "$DRY_RUN" = false ]; then
        if [ "$VERBOSE" = true ]; then
            print_warning "File already exists, skipping: $(basename "$line")"
        fi
        ((SKIP_COUNT++))
        continue
    fi
    
    # Copy file
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would copy: $line -> $TARGET_FILE"
        ((SUCCESS_COUNT++))
    else
        cp "$line" "$TARGET_FILE"
        if [ $? -eq 0 ]; then
            if [ "$VERBOSE" = true ]; then
                print_success "Copied: $(basename "$line")"
            fi
            ((SUCCESS_COUNT++))
        else
            print_error "Failed to copy: $line"
            ((ERROR_COUNT++))
        fi
    fi
    
done < "$INPUT_FILE"

# Print summary
echo ""
echo "=========================================="
print_info "Copy operation completed!"
echo "=========================================="
echo "Total files in list:     $TOTAL_FILES"
print_success "Successfully copied:     $SUCCESS_COUNT"
[ $SKIP_COUNT -gt 0 ] && print_warning "Skipped (existing):      $SKIP_COUNT"
[ $ERROR_COUNT -gt 0 ] && print_error "Errors encountered:      $ERROR_COUNT"
echo "=========================================="

# Exit with appropriate code
if [ $ERROR_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi
