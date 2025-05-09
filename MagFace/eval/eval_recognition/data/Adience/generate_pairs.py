import re
import random

# Function to process the file
def process_file(input_file, output_file):
    # Dictionary to store files grouped by their 'x' value
    x_groups = {}

    # Step 1: Build the dictionary by iterating through all lines once
    index = 0
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            match = re.match(r".*coarse_tilt_aligned_face\.(\d+)\..*\.png", line)
            if match:
                x_value = match.group(1)  # Extract 'x' value
                x_groups.setdefault(x_value, []).append(index)
            index += 1

    # Step 2: Use the dictionary to create the output
    with open(output_file, 'w') as out_file:
        for x_value, files in x_groups.items():
            for file in files:
                # 1. Start with the current file
                result = [file]
                
                # 2. Find one more file with the same x value (if possible)
                same_x_files = [f for f in x_groups[x_value] if f != file]
                if same_x_files:
                    result.append(random.choice(same_x_files))
                else:
                    result.append(file)  # Add the same file if no other exists

                # 3. Find five files with different x values
                different_x_values = [k for k in x_groups.keys() if k != x_value]
                random.shuffle(different_x_values)
                count = 0
                for other_x in different_x_values:
                    result.append(random.choice(x_groups[other_x]))
                    count += 1
                    if count == 5:
                        break
                
                # 4. Write the result to the output file
                if len(result) == 7:  # Ensure exactly 7 lines are written
                    out_file.write(f"{result[0]} {result[1]} 1\n")
                    out_file.write(f"{result[0]} {result[2]} 0\n")
                    out_file.write(f"{result[0]} {result[3]} 0\n")
                    out_file.write(f"{result[0]} {result[4]} 0\n")
                    out_file.write(f"{result[0]} {result[5]} 0\n")
                    out_file.write(f"{result[0]} {result[6]} 0\n")

# Input and output file paths
input_file = "C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/MagFace/eval/eval_recognition/data/Adience/img.list"   # Replace with your input file path
output_file = "pairs.list" # Replace with your desired output file path

# Run the function
process_file(input_file, output_file)
print(f"Results saved to {output_file}")
