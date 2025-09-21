# Import library yang diperlukan
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob

# Set up matplotlib untuk tampilan yang lebih baik
plt.rcParams['figure.figsize'] = [15, 10]

# Fungsi untuk memproses gambar dengan CLAHE
def apply_clahe(image_path, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Menerapkan CLAHE pada gambar
    
    Parameters:
    image_path: path ke file gambar
    clip_limit: batas kontras (default 2.0)
    tile_grid_size: ukuran grid tile (default 8x8)
    
    Returns:
    original_image, clahe_image
    """
    # Baca gambar
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Tidak dapat membaca gambar {image_path}")
        return None, None
    
    # Konversi BGR ke RGB untuk matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Konversi ke LAB color space untuk CLAHE yang lebih baik
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Buat objek CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Terapkan CLAHE pada L channel (luminance)
    l_clahe = clahe.apply(l_channel)
    
    # Gabungkan kembali channels
    lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])
    
    # Konversi kembali ke RGB
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return img_rgb, img_clahe

# Fungsi untuk menampilkan perbandingan gambar
def show_comparison(original, processed, title_original="Original", title_processed="CLAHE"):
    """Menampilkan perbandingan gambar sebelum dan sesudah CLAHE"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(original)
    axes[0].set_title(title_original, fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(processed)
    axes[1].set_title(title_processed, fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Fungsi untuk memproses semua gambar dalam folder
def process_all_images(input_folder, output_folder=None, clip_limit=2.0, tile_grid_size=(8,8), show_samples=3):
    """
    Memproses semua gambar dalam folder dengan CLAHE
    
    Parameters:
    input_folder: folder input gambar
    output_folder: folder output (optional)
    clip_limit: parameter CLAHE
    tile_grid_size: parameter CLAHE
    show_samples: jumlah sampel yang ditampilkan
    """
    
    # Buat folder output jika diperlukan
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Folder output: {output_folder}")
    
    # Format file gambar yang didukung
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # Kumpulkan semua file gambar
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not image_files:
        print(f"Tidak ditemukan file gambar di folder {input_folder}")
        return
    
    print(f"Ditemukan {len(image_files)} file gambar")
    print(f"Parameter CLAHE: clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")
    
    processed_count = 0
    
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        print(f"Memproses ({i+1}/{len(image_files)}): {filename}")
        
        # Terapkan CLAHE
        original, processed = apply_clahe(image_path, clip_limit, tile_grid_size)
        
        if original is None or processed is None:
            continue
        
        # Simpan hasil jika folder output disediakan
        if output_folder:
            output_path = os.path.join(output_folder, f"clahe_{filename}")
            # Konversi RGB kembali ke BGR untuk cv2.imwrite
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, processed_bgr)
        
        # Tampilkan sampel perbandingan
        if i < show_samples:
            print(f"\n--- Perbandingan untuk {filename} ---")
            show_comparison(original, processed, 
                          title_original=f"Original - {filename}", 
                          title_processed=f"CLAHE - {filename}")
        
        processed_count += 1
    
    print(f"\nSelesai! {processed_count} gambar berhasil diproses")
    if output_folder:
        print(f"Hasil disimpan di: {output_folder}")

# Fungsi untuk eksperimen dengan parameter CLAHE yang berbeda
def experiment_clahe_parameters(image_path, clip_limits=[1.0, 2.0, 3.0, 4.0], tile_sizes=[(4,4), (8,8), (16,16)]):
    """Eksperimen dengan parameter CLAHE yang berbeda"""
    
    if not os.path.exists(image_path):
        print(f"File {image_path} tidak ditemukan")
        return
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Error membaca gambar")
        return
    
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    print("Eksperimen dengan berbagai parameter CLAHE:")
    print(f"Gambar: {os.path.basename(image_path)}")
    
    # Eksperimen dengan clip limits
    fig, axes = plt.subplots(1, len(clip_limits)+1, figsize=(20, 5))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis('off')
    
    for i, clip_limit in enumerate(clip_limits):
        _, processed = apply_clahe(image_path, clip_limit=clip_limit, tile_grid_size=(8,8))
        axes[i+1].imshow(processed)
        axes[i+1].set_title(f"Clip Limit: {clip_limit}", fontsize=12)
        axes[i+1].axis('off')
    
    plt.suptitle("Perbandingan Clip Limits (Tile Size: 8x8)", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Eksperimen dengan tile sizes
    fig, axes = plt.subplots(1, len(tile_sizes)+1, figsize=(20, 5))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis('off')
    
    for i, tile_size in enumerate(tile_sizes):
        _, processed = apply_clahe(image_path, clip_limit=2.0, tile_grid_size=tile_size)
        axes[i+1].imshow(processed)
        axes[i+1].set_title(f"Tile Size: {tile_size}", fontsize=12)
        axes[i+1].axis('off')
    
    plt.suptitle("Perbandingan Tile Sizes (Clip Limit: 2.0)", fontsize=14)
    plt.tight_layout()
    plt.show()

# =============== EKSEKUSI UTAMA ===============

# 1. Cek ketersediaan folder
input_folder = "/content/images"

if not os.path.exists(input_folder):
    print(f"Error: Folder {input_folder} tidak ditemukan!")
    print("Pastikan Anda telah mengupload gambar ke folder /content/images")
else:
    print(f"Folder {input_folder} ditemukan!")
    
    # 2. Tampilkan isi folder
    files = os.listdir(input_folder)
    print(f"Isi folder ({len(files)} files):")
    for file in files[:10]:  # Tampilkan 10 file pertama
        print(f"  - {file}")
    if len(files) > 10:
        print(f"  ... dan {len(files) - 10} file lainnya")
    
    # 3. Proses semua gambar dengan parameter default
    print("\n" + "="*60)
    print("MEMULAI PEMROSESAN CLAHE")
    print("="*60)
    
    # Opsi 1: Proses tanpa menyimpan, hanya menampilkan hasil
    process_all_images(
        input_folder=input_folder,
        output_folder=None,  # Tidak menyimpan hasil
        clip_limit=2.0,
        tile_grid_size=(8,8),
        show_samples=3  # Tampilkan 3 sampel
    )
    
    # Opsi 2: Proses dan simpan hasil (uncomment jika ingin menyimpan)
    # output_folder = "/content/images_clahe"
    # process_all_images(
    #     input_folder=input_folder,
    #     output_folder=output_folder,
    #     clip_limit=2.0,
    #     tile_grid_size=(8,8),
    #     show_samples=3
    # )
    
    # 4. Eksperimen dengan parameter (opsional)
    print("\n" + "="*60)
    print("EKSPERIMEN PARAMETER CLAHE")
    print("="*60)
    
    # Ambil gambar pertama untuk eksperimen
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if image_files:
        experiment_image = image_files[0]
        print(f"Menggunakan gambar: {os.path.basename(experiment_image)}")
        
        # Uncomment untuk menjalankan eksperimen parameter
        # experiment_clahe_parameters(experiment_image)

print("\n" + "="*60)
print("TIPS PENGGUNAAN:")
print("="*60)
print("1. Clip Limit (1.0-4.0): Semakin tinggi = kontras lebih kuat")
print("2. Tile Grid Size: (4,4)=detail halus, (16,16)=area luas")
print("3. Untuk gambar gelap: clip_limit=3.0-4.0")
print("4. Untuk gambar terang: clip_limit=1.0-2.0")
print("5. Uncomment bagian yang diinginkan untuk menyimpan hasil")
