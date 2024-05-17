import os
import gdown

def download_files():
    # Google Drive file IDs
    file_ids = {
        "final_merged": "1IYEQfBajVw6qHbw_B-tYBC6Dq6wIGJhj",
        "members": "1H3jKx_NDiHR10ETPm_BoAgcy6iSR49JN",
        "test": "1nmQgqXg-gLyiCSAnFGm4Qa4-H1kxcvUM",
        "train": "1aDcGlF7v51PunxRTbDoWaPo49-oHDTc_",
        "transactions": "1u8R118HgBJaM1Sg1q5TDr-t4iaOuUHjd",
        "user_logs": "1WE7o89ENnIuzGDskN0CgviNjifv6rRDu"

    }

    # Corresponding local file paths
    local_paths = {
        "final_merged": "./data/final_merged.csv",
        "members": "./data/members.csv",
        "test": "./data/test.csv",
        "train": "./data/train.csv",
        "transactions": "./data/transactions.csv",
        "user_logs": "./data/user_logs.csv"
    }

    os.makedirs('./data', exist_ok=True)

    # Download each file from Google Drive
    for key, file_id in file_ids.items():
        file_url = f'https://drive.google.com/uc?id={file_id}'
        local_path = local_paths[key]
        gdown.download(file_url, local_path, quiet=False)

if __name__ == "__main__":
    download_files()
