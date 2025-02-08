import os
import time
from fabric import Connection
import argparse
from dagshub import upload_files
ClusterHOST = "gpu.scholar.rcac.purdue.edu"
ClusterUser = "tyalaman"
ClusterKey = "/Users/tejas/.ssh/id_rsa"

# BUCKET_NAME = "skunkworks"
# ENDPOINT_URL = "https://dagshub.com/api/v1/repo-buckets/s3/YalmanchiliTejas"
# ACCESS_TOKEN = "34119164be000c31552b19b34fc8c9ac18779dfc"  # Replace with your DagsHub access token
# REGION = "us-east-1"
# 
username="YalmanchiliTejas"
repo_name="skunkworks"

def submit_and_monitor_slurm():
    # Create a new connection
    c = Connection(host=ClusterHOST, user=ClusterUser, connect_kwargs={'key_filename': ClusterKey})

    result = c.run("cd /home/tyalaman/skunkworks/deformations/slurm_scripts && sbatch pre_process_script.sh", hide=True)
    print("Job Submitted")
    job_id = result.stdout.split(" ")[-1].strip()
    print("Monitorung Job")
    while True:
        result = c.run("squeue -u tyalaman", hide=True)
        print(result.stdout)
        if  job_id not in result.stdout:
            break
        time.sleep(30)
    print("Job Finished")
def upload_to_dagshub(directory):
    
    # Upload to DagsHub storage bucket
    upload_files(
        f"{username}/{repo_name}",
        directory,
        remote_path="images",
        bucket=True
    )
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments to run the script or upload data')
    parser.add_argument('--upload', type=str, help='Upload the images to DagsHub')
    
    args = parser.parse_args()
    if args.upload:
        upload_to_dagshub(args.upload)
    else: 
        submit_and_monitor_slurm()
   


