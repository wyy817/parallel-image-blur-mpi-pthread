def read_pgm(filename):
    """Read PGM file and return width, height, and pixel data"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Remove empty lines and whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        # Verify format
        if not lines[0] == 'P2':
            raise ValueError(f'Invalid format: {lines[0]}')
        
        # Get dimensions from second line
        width, height = map(int, lines[1].split())
        maxval = int(lines[2])
        
        # Read pixel data
        pixels = []
        current_row = []
        
        for line in lines[3:]:
            values = [int(x) for x in line.split()]
            current_row.extend(values)
            
            # When we have a complete row, verify and store it
            while len(current_row) >= width:
                pixels.extend(current_row[:width])
                current_row = current_row[width:]
        
        # Verify total pixel count
        if len(pixels) != width * height:
            raise ValueError(f'Invalid pixel count: expected {width * height}, got {len(pixels)}')
            
        return width, height, pixels
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        raise

def print_region(pixels, width, x, y, size=5):
    """Print a region of pixels around a specific point"""
    start_y = max(0, y - size)
    end_y = min(y + size + 1, len(pixels) // width)
    start_x = max(0, x - size)
    end_x = min(x + size + 1, width)
    
    print(f"\nRegion around ({x},{y}):")
    print("    " + "".join(f"{i:4}" for i in range(start_x, end_x)))
    for y_pos in range(start_y, end_y):
        print(f"{y_pos:3}", end=" ")
        for x_pos in range(start_x, end_x):
            val = pixels[y_pos * width + x_pos]
            print(f"{val:4}", end="")
        print()

def compare_pgm_files(file1, file2, tolerance=0):
    """Compare two PGM files pixel by pixel"""
    try:
        # Read files
        width1, height1, pixels1 = read_pgm(file1)
        width2, height2, pixels2 = read_pgm(file2)
        
        # Basic validation
        if width1 != width2 or height1 != height2:
            print(f"Size mismatch: {width1}x{height1} vs {width2}x{height2}")
            return False
        
        # Compare pixels
        differences = []
        diff_count = 0
        max_diff = 0
        total_diff = 0
        
        min_val1, max_val1 = min(pixels1), max(pixels1)
        min_val2, max_val2 = min(pixels2), max(pixels2)
        
        print(f"\nFile statistics:")
        print(f"File1 value range: {min_val1} - {max_val1}")
        print(f"File2 value range: {min_val2} - {max_val2}")
        
        # Count differences by magnitude
        diff_histogram = {1: 0, 2: 0, '3-5': 0, '6-10': 0, '11+': 0}
        
        for i, (p1, p2) in enumerate(zip(pixels1, pixels2)):
            diff = abs(p1 - p2)
            if diff > tolerance:
                y = i // width1
                x = i % width1
                diff_count += 1
                
                # Update histogram
                if diff == 1: diff_histogram[1] += 1
                elif diff == 2: diff_histogram[2] += 1
                elif diff <= 5: diff_histogram['3-5'] += 1
                elif diff <= 10: diff_histogram['6-10'] += 1
                else: diff_histogram['11+'] += 1
                
                # Store large differences for detailed review
                if diff > tolerance:
                    differences.append((x, y, p1, p2, diff))
                
                max_diff = max(max_diff, diff)
                total_diff += diff
        
        # Report results
        total_pixels = width1 * height1
        diff_percentage = (diff_count / total_pixels) * 100
        
        print(f"\nComparison results:")
        print(f"Total pixels: {total_pixels:,}")
        print(f"Different pixels: {diff_count:,} ({diff_percentage:.2f}%)")
        
        if diff_count > 0:
            print(f"\nDifference histogram:")
            for key, count in diff_histogram.items():
                print(f"Diff {key}: {count:,} pixels")
            
            print(f"\nMaximum difference: {max_diff}")
            print(f"Average difference: {total_diff / diff_count:.2f}")
            
            differences.sort(key=lambda x: x[4], reverse=True)
            print("\nLargest differences:")
            print("Position(x,y)\tFile1\tFile2\tDiff")
            for x, y, p1, p2, diff in differences[:5]:
                print(f"({x},{y})\t\t{p1}\t{p2}\t{diff}")
                print_region(pixels1, width1, x, y)
                print_region(pixels2, width2, x, y)
                
        else:
            print("Images are identical!")
            
        return diff_count == 0
            
    except Exception as e:
        print(f"Error during comparison: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python3 compare_pgm.py <file1> <file2> [tolerance]")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    tolerance = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    compare_pgm_files(file1, file2, tolerance)