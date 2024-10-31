import os

# aws cli 
class S3Sync:
    # syncs local folder directory content to aws bucket
    # sends data from local directory folder---> aws s3 
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        command = f"aws s3 sync {folder} {aws_bucket_url}"
        os.system(command)

    # syncs aws bucket content with local directory folder 
    # sends data aws s3 --> local directory folder
    def sync_folder_from_s3(self, folder, aws_bucket_url):
        command = f"aws s3 sync {aws_bucket_url} {folder}"
        
