import subprocess
import time
import csv

# Function to get GPU power usage
def get_gpu_power():
    try:
        # Execute nvidia-smi command to get the power draw
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        power = result.stdout.strip()
        return power
    except Exception as e:
        print(f"Error collecting GPU power data: {e}")
        return None

# Function to run a single application and collect power data
def run_application(app_name):
    # Start the process for the given application
    process = subprocess.Popen([f'./{app_name}'])
    power_measurements = []

    # Initialize a timer
    start_time = time.time()

    # Collect GPU power until the process terminates
    while process.poll() is None:
        current_time = time.time() - start_time  # Calculate elapsed time
        power = get_gpu_power()
        if power:
            power_measurements.append((current_time, power))
        # Sleep for a short interval (e.g., 0.1 second)
        time.sleep(0.1)

    # Process has finished, return power data
    return power_measurements

# Main function to run the applications and collect GPU power data
def main():
    # Run v1 and collect its GPU power data
    power_data_v1 = run_application('v2')
    # Write results to CSV for v1
    with open('gpu_power_results_v2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'GPU Power (W)'])
        for time_elapsed, power in power_data_v1:
            writer.writerow([f"{time_elapsed:.2f}", power])

if __name__ == "__main__":
    main()

