import cv2
import tkinter as tk
from tkinter import filedialog

def select_image():
    """Mở hộp thoại chọn file ảnh"""
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    root.attributes('-topmost', True)  # Đưa dialog lên trên cùng
    
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh CCCD",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def crop_portrait(img):
    """Crop phần chân dung từ ảnh CCCD"""
    h, w = img.shape[:2]
    
    # Tỷ lệ crop (có thể điều chỉnh)
    y_start = int(0.35 * h)
    y_end = int(0.95 * h)
    x_start = int(0.03 * w)
    x_end = int(0.29 * w)
    
    # Crop
    portrait = img[y_start:y_end, x_start:x_end]
    
    return portrait, (x_start, y_start, x_end, y_end)

def main():
    print("=" * 60)
    print(" CROP CHÂN DUNG TỪ ẢNH CCCD ".center(60, "="))
    print("=" * 60)
    
    # Chọn ảnh
    print("\nMở hộp thoại chọn file...")
    path = select_image()
    
    if not path:
        print("Không chọn file nào. Thoát chương trình.")
        return
    
    print(f"Đã chọn: {path}")
    
    # Đọc ảnh
    img = cv2.imread(path)
    
    if img is None:
        print("Không đọc được ảnh. Kiểm tra lại file.")
        return
    
    h, w = img.shape[:2]
    print(f"Kích thước ảnh gốc: {w}x{h} pixels")
    
    # Crop chân dung
    portrait, coords = crop_portrait(img)
    x_start, y_start, x_end, y_end = coords
    
    # Lưu kết quả
    output_path = "cccd_portrait.jpg"
    cv2.imwrite(output_path, portrait)
    print(f"\nĐã lưu ảnh chân dung: {output_path}")
    
    # Thông tin crop
    print("\n" + "-" * 60)
    print("THÔNG TIN VÙNG CROP:")
    print("-" * 60)
    print(f"Tọa độ X: {x_start} → {x_end} (rộng: {x_end - x_start}px)")
    print(f"Tọa độ Y: {y_start} → {y_end} (cao: {y_end - y_start}px)")
    print(f"Kích thước ảnh crop: {portrait.shape[1]}x{portrait.shape[0]} pixels")
    
    # Hiển thị
    print("\nNhấn phím bất kỳ để đóng cửa sổ...")
    cv2.imshow("Chân dung đã crop", portrait)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(" HOÀN TẤT ".center(60, "="))
    print("=" * 60)

if __name__ == '__main__':
    main()