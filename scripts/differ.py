
def compare_files(file1_path, file2_path):
    with open(file1_path, 'r', encoding='utf-8') as f1, \
         open(file2_path, 'r', encoding='utf-8') as f2:
        
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        if len(lines1) != len(lines2):
            print(f"Files have different number of lines: {len(lines1)} vs {len(lines2)}")
            
        for i, (line1, line2) in enumerate(zip(lines1, lines2), 1):
            if line1 != line2:
                print(f"Difference at line {i}:")
                print(f"File 1: {line1.strip()}")
                print(f"File 2: {line2.strip()}")
                print()

# Usage:
compare_files('isotra_2023.html', 'output2.html')