import cv2
import tkinter as tk
from tkinter import filedialog

def select_image():
    """Mở hộp thoại chọn file ảnh"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh CCCD",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def crop_text_region(img):
    """Crop phần text từ ảnh CCCD"""
    h, w = img.shape[:2]
    
    # Tham chiếu phần hình đã cắt
    x_end_hinh = int(0.28 * w)
    
    # Tọa độ crop phần text
    y_start = int(0.40 * h)
    y_end = int(0.98 * h)
    x_start_chu = x_end_hinh + 10
    x_end_chu = int(0.95 * w)
    
    # Crop
    text_region = img[y_start:y_end, x_start_chu:x_end_chu]
    
    return text_region, (x_start_chu, y_start, x_end_chu, y_end)

def main():
    print("=" * 60)
    print(" CROP VÙNG TEXT TỪ ẢNH CCCD ".center(60, "="))
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
    
    # Crop vùng text
    text_region, coords = crop_text_region(img)
    x_start_chu, y_start, x_end_chu, y_end = coords
    
    # Lưu kết quả
    output_path = "cccd_text.jpg"
    cv2.imwrite(output_path, text_region)
    print(f"\nĐã lưu vùng text: {output_path}")
    
    # Thông tin crop
    print("\n" + "-" * 60)
    print("THÔNG TIN VÙNG TEXT:")
    print("-" * 60)
    print(f"Tọa độ X: {x_start_chu} → {x_end_chu} (rộng: {x_end_chu - x_start_chu}px)")
    print(f"Tọa độ Y: {y_start} → {y_end} (cao: {y_end - y_start}px)")
    print(f"Kích thước vùng text: {text_region.shape[1]}x{text_region.shape[0]} pixels")
    
    # Hiển thị
    print("\nNhấn phím bất kỳ để đóng cửa sổ...")
    cv2.imshow("Vùng text đã crop", text_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(" HOÀN TẤT ".center(60, "="))
    print("=" * 60)

if __name__ == '__main__':
    main()