
def download_datasets():
    savedir = 'Datasets'    
    os.makedirs(savedir,exist_ok=True)
    os.system(f'kaggle datasets download -p {savedir} -d grassknoted/asl-alphabet --unzip')

