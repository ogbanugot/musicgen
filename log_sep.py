import os

# Define the directory containing the files
directory = "vocals"

# Define the log file
log_file = "separated.txt"

# Open the log file in write mode
with open(log_file, "w") as log:
    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        # Split the filename by "_(Vocals)_"
        parts = filename.split("_(Vocals)_")
        part = parts[0]
        # Append ".wav" to each part and write to the log file
        #for part in parts:
        log.write(f"{part}\n")

print(f"Filenames have been processed and logged into {log_file}.")