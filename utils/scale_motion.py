import csv

def scale_motion(input_file, output_file, scale_factor):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            if row[0].startswith('#') or row[0].startswith('//'):
                # Copy header or comment lines as is
                writer.writerow(row)
            else:
                # Scale joint angles
                scaled_row = row[:2] + [str(float(value) * scale_factor) for value in row[2:]]
                writer.writerow(scaled_row)

# Scale TurnLeft60.motion to TurnLeft10.motion
scale_motion('motions/TurnRight60.motion', 'motions/TurnRight10.motion', scale_factor=10/60)