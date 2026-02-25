import os
import shutil

def download_datasets():
    savedir = 'Datasets/raw'
    os.makedirs(savedir, exist_ok=True)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print("=====================================================")
        print("1. ĐANG Kiểm Tra: Bộ ASL Alphabet gốc (Grassknoted)")
        print("=====================================================")
        if not os.path.exists(f'{savedir}/asl_alphabet_train'):
            print("Chưa có Data, tiến hành tải...")
            api.dataset_download_cli('grassknoted/asl-alphabet', path=savedir, unzip=True)
            # Kaggle có thể giải nén lồng, thư mục gốc là asl_alphabet_train
        else:
            print("[BỎ QUA] Đã có sẵn Bộ ASL Alphabet gốc.")

        print("=====================================================")
        print("2. ĐANG Kiểm Tra: Bộ ASL Alphabet Test (Danielenricocahall)")
        print("=====================================================")
        if not os.path.exists(f'{savedir}/asl_alphabet_test_real_bg') and not os.path.exists(f'{savedir}/asl_alphabet_test'):
            print("Chưa có Data, tiến hành tải...")
            api.dataset_download_cli('danielenricocahall/asl-alphabet-test', path=savedir, unzip=True)
            if os.path.exists(f'{savedir}/asl-alphabet-test'):
                try:
                    os.rename(f'{savedir}/asl-alphabet-test', f'{savedir}/asl_alphabet_test_real_bg')
                except:
                    pass
        else:
            print("[BỎ QUA] Đã có sẵn Bộ ASL Alphabet Test.")

        print("=====================================================")
        print("3. ĐANG Kiểm Tra: Bộ Sign Language MNIST (Datamunge)")
        print("=====================================================")
        if not os.path.exists(f'{savedir}/sign_mnist_train'): # Thường file CSV sẽ nằm ở đây
            print("Chưa có Data, tiến hành tải MNIST...")
            api.dataset_download_cli('datamunge/sign-language-mnist', path=savedir, unzip=True)
        else:
            print("[BỎ QUA] Đã có sẵn Bộ MNIST.")

        print("\n✅ TẢI DATA TỰ ĐỘNG HOÀN TẤT!\n")
    except Exception as e:
        print(f"[LỖI KAGGLE] Không thể tải dữ liệu: {e}")
        print("👉 HƯỚNG DẪN FIX: Bạn VUI LÒNG KIỂM TRA LẠI xem đã dán file `kaggle.json` vào đúng `C:\\Users\\admin\\.kaggle\\kaggle.json` chưa nhé!")

if __name__ == "__main__":
    download_datasets()
