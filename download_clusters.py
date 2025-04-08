import gdown
import os

# Google Drive folder URL
folder_url = "https://drive.google.com/drive/folders/13gv6EkHwD7L7FkS3qI9iIyHVYlQDxaIP"

# Create a directory to store downloaded files
output_dir = "downloaded_clusters"
os.makedirs(output_dir, exist_ok=True)

# Download all files from the folder
gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)

print(f"All files have been downloaded to the '{output_dir}' directory.") 