# -*- coding: utf-8 -*-
"""
OCR CCCD - Phiên bản cải tiến với spell correction
- Sửa lỗi chính tả từ OCR
- Tách các trường bị dính nhau
- Trích xuất đầy đủ tất cả thông tin
- Cho phép người dùng chọn file ảnh
"""
import os, re, cv2, pytesseract, numpy as np
from pytesseract import Output
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog

# ========== CẤU HÌNH ==========
TESS_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_PREFIX = r"C:\Program Files\Tesseract-OCR\tessdata"

pytesseract.pytesseract.tesseract_cmd = TESS_CMD
os.environ['TESSDATA_PREFIX'] = TESSDATA_PREFIX

# ========== BẢNG MAP CHO SỐ ==========
DIGIT_MAP = {
    'O': '0', 'o': '0', 'Q': '0', 'D': '0',
    'I': '1', 'l': '1', '|': '1', '!': '1', 'i': '1', 'J': '1',
    'Z': '2', 'z': '2',
    'E': '3', 'e': '3',
    'A': '4', 'h': '4',
    'S': '5', 's': '5',
    'G': '6', 'b': '6',
    'T': '7', 't': '7',
    'B': '8', 'R': '8',
    'g': '9', 'q': '9', 'P': '9'
}

# ========== SPELL CORRECTION MAP ==========
SPELL_CORRECTIONS = {
    'sá': 'Số', 'só': 'Số', 'số': 'Số', 'sô': 'Số',
    'Họ và tên': 'Họ và tên', 'Họ va tên': 'Họ và tên', 'Ho va ten': 'Họ và tên',
    'Ngày sinh': 'Ngày sinh', 'Ngay sinh': 'Ngày sinh', 'Ngáy sinh': 'Ngày sinh',
    'Dateofbirth': 'Ngày sinh', 'Date of birth': 'Ngày sinh',
    'Giớitính': 'Giới tính', 'Gioi tinh': 'Giới tính', 'Giói tính': 'Giới tính',
    'Quốctịch': 'Quốc tịch', 'Quoc tich': 'Quốc tịch', 'Quốc tich': 'Quốc tịch',
    'Nationality': 'Quốc tịch', 'Nafionalty': 'Quốc tịch', 'Nationlity': 'Quốc tịch',
    'Quê quán': 'Quê quán', 'Que quan': 'Quê quán', 'Quê quan': 'Quê quán',
    'Nời thường trú': 'Nơi thường trú', 'Noi thuong tru': 'Nơi thường trú',
    'Nơi thuờng trú': 'Nơi thường trú', 'Nơi thường tru': 'Nơi thường trú',
    'Địa chỉ': 'Nơi thường trú', 'Dia chi': 'Nơi thường trú'
}

def correct_spelling(text):
    """Sửa lỗi chính tả cơ bản"""
    if not text:
        return text
    
    result = text
    for wrong, correct in SPELL_CORRECTIONS.items():
        result = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, result, flags=re.IGNORECASE)
    
    return result

def normalize_to_digits(s):
    """Chuyển đổi chuỗi có thể nhầm thành số"""
    if not s:
        return ""
    result = ''.join(DIGIT_MAP.get(ch, ch) for ch in s)
    result = re.sub(r'[^0-9]', '', result)
    return result

# ========== TIỆN ÍCH ==========
def safe_mean(values):
    """Tính trung bình an toàn"""
    valid = [v for v in values if v >= 0]
    return float(np.mean(valid)) if valid else 0.0

def count_vietnamese_chars(text):
    """Đếm ký tự tiếng Việt có dấu"""
    if not text:
        return 0
    vietnamese_chars = (
        'àáạảãâầấậẩẫăằắặẳẵ'
        'èéẹẻẽêềếệểễ'
        'ìíịỉĩ'
        'òóọỏõôồốộổỗơờớợởỡ'
        'ùúụủũưừứựửữ'
        'ỳýỵỷỹ'
        'đ'
    )
    vietnamese_chars += vietnamese_chars.upper()
    return sum(1 for c in text if c in vietnamese_chars)

# ========== CHỌN FILE ẢNH ==========
def select_image_file():
    """Mở hộp thoại chọn file ảnh"""
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh CCCD",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

# ========== TIỀN XỬ LÝ ẢNH ==========
def preprocess_ultimate(img):
    """Tiền xử lý tối ưu nhất cho CCCD"""
    if img is None:
        raise ValueError("Ảnh đầu vào là None")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape
    print(f"Kích thước: {w}x{h} pixels")
    
    # 1. Denoise nhẹ
    denoised = cv2.fastNlMeansDenoising(gray, h=4, templateWindowSize=7, searchWindowSize=21)
    
    # 2. Tăng độ sắc nét (unsharp mask)
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    
    # 3. CLAHE nhẹ
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    contrast = clahe.apply(sharpened)
    
    # 4. Adaptive threshold
    block_size = 25 if h < 800 else 31
    adaptive = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, 2
    )
    
    print(f"   ✓ Hoàn tất (block_size={block_size})")
    
    return adaptive, contrast

# ========== OCR TỐI ƯU ==========
def ocr_ultimate(img_binary, img_gray):
    """OCR với nhiều config và chọn tốt nhất"""
    results = []
    
    configs = [
        ('VIE_PSM6_Binary', '--oem 3 --psm 6 -l vie', img_binary),
        ('VIE_PSM4_Binary', '--oem 3 --psm 4 -l vie', img_binary),
        ('VIE+ENG_PSM6_Binary', '--oem 3 --psm 6 -l vie+eng', img_binary),
        ('VIE+ENG_PSM4_Binary', '--oem 3 --psm 4 -l vie+eng', img_binary),
        ('VIE_PSM6_Gray', '--oem 3 --psm 6 -l vie', img_gray),
    ]
    
    print(f"\n   Đang chạy {len(configs)} config OCR...")
    
    for idx, (name, cfg, img_use) in enumerate(configs, 1):
        try:
            data = pytesseract.image_to_data(img_use, config=cfg, output_type=Output.DICT)
            raw_text = pytesseract.image_to_string(img_use, config=cfg)
            
            # Sửa lỗi chính tả
            raw_text = correct_spelling(raw_text)
            
            lines = rebuild_lines_smart(data)
            
            avg_conf = safe_mean([l['conf'] for l in lines])
            viet_count = count_vietnamese_chars(raw_text)
            text_length = len(raw_text.strip())
            
            score = avg_conf * 1.0 + viet_count * 5.0 + min(text_length / 100.0, 15.0)
            
            results.append({
                'name': name,
                'raw_text': raw_text,
                'lines': lines,
                'avg_conf': avg_conf,
                'viet_count': viet_count,
                'score': score,
                'data': data
            })
            
            print(f"      [{idx}/{len(configs)}] {name:25s} | Conf: {avg_conf:5.1f} | Score: {score:6.1f}")
            
        except Exception as e:
            print(f"      [{idx}/{len(configs)}] {name:25s} | ❌ Lỗi: {e}")
            continue
    
    if not results:
        raise Exception("Tất cả config OCR đều thất bại!")
    
    best = max(results, key=lambda x: x['score'])
    print(f"\n   🏆 Config tốt nhất: {best['name']}")
    
    return best, results

def rebuild_lines_smart(data):
    """Rebuild lines với khoảng trắng chính xác"""
    n = len(data.get('text', []))
    lines_dict = defaultdict(list)
    
    for i in range(n):
        word = (data['text'][i] or '').strip()
        if not word:
            continue
        
        # Sửa chính tả từng word
        word = correct_spelling(word)
        
        try:
            conf = float(data['conf'][i])
        except:
            conf = 0.0
        
        block = data.get('block_num', [0]*n)[i]
        par = data.get('par_num', [0]*n)[i]
        line = data.get('line_num', [0]*n)[i]
        word_num = data.get('word_num', [0]*n)[i]
        left = data.get('left', [0]*n)[i]
        width = data.get('width', [0]*n)[i]
        
        key = (block, par, line)
        lines_dict[key].append({
            'word': word,
            'conf': conf,
            'word_num': word_num,
            'left': left,
            'right': left + width
        })
    
    result_lines = []
    for key in sorted(lines_dict.keys()):
        words_list = sorted(lines_dict[key], key=lambda x: (x['word_num'], x['left']))
        
        # Giữ words có conf >= 3 hoặc có ký tự Việt
        filtered_words = []
        for w in words_list:
            if w['conf'] >= 3 or count_vietnamese_chars(w['word']) > 0:
                filtered_words.append(w)
        
        if not filtered_words:
            filtered_words = words_list
        
        line_text = ' '.join([w['word'] for w in filtered_words])
        line_conf = safe_mean([w['conf'] for w in filtered_words])
        
        if line_text.strip():
            result_lines.append({
                'text': line_text,
                'conf': line_conf
            })
    
    return result_lines

# ========== TRÍCH XUẤT CẢI TIẾN ==========
def extract_id_number(lines, raw_text):
    """Trích xuất số CCCD (12 chữ số) hoặc CMND (9 chữ số)"""
    full_text = "\n".join([l['text'] for l in lines]) + "\n" + raw_text
    
    # Tìm dòng có "Số" hoặc "ID"
    id_lines = []
    for i, line in enumerate(lines):
        if re.search(r'(số|so|id|cccd|cmnd|no\.)', line['text'], re.IGNORECASE):
            id_lines.append(line['text'])
            if i + 1 < len(lines):
                id_lines.append(lines[i + 1]['text'])
    
    search_text = "\n".join(id_lines) if id_lines else full_text
    
    # Tìm 12 chữ số (CCCD mới) hoặc 9 chữ số (CMND cũ)
    patterns = [r'\b\d{12}\b', r'\b\d{9}\b']
    for pattern in patterns:
        matches = re.findall(pattern, search_text)
        if matches:
            return matches[0]
    
    # Tìm số có thể bị nhầm
    tokens = re.findall(r'[0-9OIlQDSsBgJAhEeTtRPq|!iz]{9,15}', search_text)
    for token in tokens:
        normalized = normalize_to_digits(token)
        if len(normalized) in [9, 12]:
            return normalized
    
    return None

def extract_name(lines):
    """Trích xuất họ tên"""
    for i, line in enumerate(lines):
        text = line['text']
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['họ và tên', 'ho va ten', 'full name', 'name']):
            # Lấy dòng tiếp theo
            if i + 1 < len(lines):
                name = lines[i + 1]['text'].strip()
                name = re.sub(r'\s*:\s*$', '', name)
                if name and name.isupper() and not re.search(r'\d', name):
                    return name
            
            # Lấy sau dấu :
            if ':' in text:
                name = text.split(':', 1)[1].strip()
                name = re.sub(r'\s*:\s*$', '', name)
                if name and not re.search(r'\d', name):
                    return name
    
    # Fallback: dòng viết hoa không có số
    for line in lines:
        text = line['text'].strip()
        text = re.sub(r'\s*:\s*$', '', text)
        if text and text.isupper() and not re.search(r'\d', text):
            words = text.split()
            if 2 <= len(words) <= 5:
                return text
    
    return None

def extract_dob(lines, raw_text):
    """Trích xuất ngày sinh"""
    full_text = "\n".join([l['text'] for l in lines]) + "\n" + raw_text
    
    # Tìm dòng có "Ngày sinh"
    dob_lines = []
    for i, line in enumerate(lines):
        if re.search(r'(ngày sinh|ngay sinh|date.*birth|dob)', line['text'], re.IGNORECASE):
            dob_lines.append(line['text'])
            if i + 1 < len(lines):
                dob_lines.append(lines[i + 1]['text'])
    
    search_text = "\n".join(dob_lines) if dob_lines else full_text
    
    # Pattern: dd/mm/yyyy hoặc dd-mm-yyyy
    patterns = [
        r'(\d{1,2})\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{4})',
        r'(\d{1,2})\s*[\.\s]\s*(\d{1,2})\s*[\.\s]\s*(\d{4})'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, search_text)
        for match in matches:
            try:
                day = int(re.sub(r'\s', '', match[0]))
                month = int(re.sub(r'\s', '', match[1]))
                year = int(re.sub(r'\s', '', match[2]))
                
                if 1 <= day <= 31 and 1 <= month <= 12 and 1920 <= year <= 2025:
                    return f"{day:02d}/{month:02d}/{year:04d}"
            except:
                continue
    
    # Fallback với normalize
    date_tokens = re.findall(r'[\dOIl\s/.\-]{8,20}', search_text)
    for token in date_tokens:
        mapped = ''.join(DIGIT_MAP.get(ch, ch) if ch in DIGIT_MAP else ch for ch in token)
        
        for pattern in patterns:
            match = re.search(pattern, mapped)
            if match:
                try:
                    day = int(match.group(1))
                    month = int(match.group(2))
                    year = int(match.group(3))
                    
                    if 1 <= day <= 31 and 1 <= month <= 12 and 1920 <= year <= 2025:
                        return f"{day:02d}/{month:02d}/{year:04d}"
                except:
                    continue
    
    return None

def extract_sex(lines, raw_text):
    """Trích xuất giới tính"""
    full_text = " ".join([l['text'] for l in lines]) + " " + raw_text
    full_text_lower = full_text.lower()

    patterns = [
        r'giới tính\s*:\s*(nam|nữ|nu|male|female)\b',
        r'sex\s*:\s*(nam|nữ|nu|male|female)\b',
        r'gender\s*:\s*(nam|nữ|nu|male|female)\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, full_text_lower, re.IGNORECASE)
        if match:
            sex_value = match.group(1).strip().lower()
            if sex_value in ['nam', 'male']:
                return 'Nam'
            elif sex_value in ['nữ', 'nu', 'female']:
                return 'Nữ'

    # Fallback
    if 'nam' in full_text_lower and 'việt nam' not in full_text_lower[:full_text_lower.find('nam') + 3]:
        return 'Nam'
    elif 'nữ' in full_text_lower or 'nu' in full_text_lower:
        return 'Nữ'
    elif 'male' in full_text_lower:
        return 'Nam'
    elif 'female' in full_text_lower:
        return 'Nữ'

    return None

def extract_nation(lines, raw_text):
    """Trích xuất quốc tịch"""
    full_text = " ".join([l['text'] for l in lines]) + " " + raw_text
    full_text_lower = full_text.lower()

    patterns = [
        r'quốc tịch\s*:\s*(việt nam|viet nam|vietnam|vn|việtnam)\b',
        r'nationality\s*:\s*(việt nam|viet nam|vietnam|vn|việtnam)\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, full_text_lower, re.IGNORECASE)
        if match:
            return 'Việt Nam'

    if 'việt nam' in full_text_lower or 'viet nam' in full_text_lower or 'vietnam' in full_text_lower:
        return 'Việt Nam'

    for i, line in enumerate(lines):
        text = line['text']
        text_lower = text.lower()
        
        if 'quốc tịch' in text_lower or 'nationality' in text_lower:
            if ':' in text:
                parts = text.split(':', 1)
                if len(parts) == 2:
                    nation = parts[1].strip()
                    nation = re.sub(r'[^\w\sÀ-ỹ]', ' ', nation).strip()
                    if nation and len(nation) >= 3:
                        return nation.capitalize()
            
            if i + 1 < len(lines):
                next_text = lines[i + 1]['text'].strip()
                if 'việt nam' in next_text.lower() or 'viet nam' in next_text.lower():
                    return 'Việt Nam'
                elif next_text and len(next_text) >= 3 and not re.search(r'\d{3,}', next_text):
                    return next_text.capitalize()
    
    return None

def extract_origin(lines):
    """Trích xuất quê quán"""
    for i, line in enumerate(lines):
        text = line['text']
        text_lower = text.lower()
        
        if 'quê quán' in text_lower or 'place of origin' in text_lower:
            if ':' in text:
                parts = text.split(':', 1)
                if len(parts) == 2:
                    origin = parts[1].strip()
                    if origin and len(origin) > 5:
                        return origin
            
            if i + 1 < len(lines):
                next_text = lines[i + 1]['text'].strip()
                if next_text and len(next_text) > 5:
                    return next_text
    
    return None

def extract_address(lines):
    """Trích xuất nơi thường trú"""
    address_parts = []
    
    for i, line in enumerate(lines):
        text = line['text']
        text_lower = text.lower()
        
        keywords = ['nơi thường trú', 'place of residence', 'thường trú', 'địa chỉ', 'noi thuong tru']
        
        if any(kw in text_lower for kw in keywords):
            if ':' in text:
                parts = text.split(':', 1)
                if len(parts) == 2:
                    addr = parts[1].strip()
                    if addr and len(addr) > 3:
                        address_parts.append(addr)
            
            for j in range(i + 1, min(i + 10, len(lines))):
                next_text = lines[j]['text'].strip()
                
                stop_keywords = ['ngày', 'date', 'có giá trị', 'valid', 'giới tính']
                if any(kw in next_text.lower() for kw in stop_keywords):
                    break
                
                if next_text and len(next_text) > 3:
                    if not re.match(r'^\d+$', next_text):
                        if re.search(r'[a-zA-ZÀ-ỹ]', next_text):
                            address_parts.append(next_text)
            
            break
    
    if address_parts:
        full_address = ', '.join(address_parts)
        full_address = re.sub(r',\s*,', ',', full_address)
        full_address = re.sub(r'\s+', ' ', full_address).strip(' ,')
        
        if len(full_address) >= 10:
            return full_address
    
    return None

# ========== MAIN ==========
def main():
    print("=" * 80)
    print(" OCR CCCD - TRÍCH XUẤT ĐẦY ĐỦ & CHÍNH XÁC ".center(80, "="))
    print("=" * 80)
    
    # Chọn file ảnh
    print("\n📂 Chọn file ảnh...")
    image_path = select_image_file()
    if not image_path:
        raise SystemExit("❌ Không chọn file ảnh nào!")
    
    # Đọc ảnh
    print(f"\n📂 Đọc ảnh: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"❌ Không đọc được: {image_path}")
    
    print(f"✓ Thành công: {img.shape[1]}x{img.shape[0]} px")
    
    # Tiền xử lý
    print("\n🔧 Tiền xử lý ảnh...")
    img_binary, img_gray = preprocess_ultimate(img)
    
    cv2.imwrite("debug_final_binary.png", img_binary)
    cv2.imwrite("debug_final_gray.png", img_gray)
    print("✓ Lưu: debug_final_binary.png, debug_final_gray.png")
    
    # OCR
    print("\n🔍 OCR...")
    best_result, all_results = ocr_ultimate(img_binary, img_gray)
    
    # Hiển thị dòng đã sửa lỗi
    print("\n" + "=" * 80)
    print(" CÁC DÒNG SAU KHI SỬA LỖI CHÍNH TẢ ".center(80, "="))
    print("=" * 80)
    for idx, line in enumerate(best_result['lines'], 1):
        print(f"[{idx:2d}] [{line['conf']:5.1f}] {line['text']}")
    
    # Trích xuất
    print("\n" + "=" * 80)
    print(" TRÍCH XUẤT THÔNG TIN ".center(80, "="))
    print("=" * 80)
    
    id_num = extract_id_number(best_result['lines'], best_result['raw_text'])
    name = extract_name(best_result['lines'])
    dob = extract_dob(best_result['lines'], best_result['raw_text'])
    sex = extract_sex(best_result['lines'], best_result['raw_text'])
    nation = extract_nation(best_result['lines'], best_result['raw_text'])
    origin = extract_origin(best_result['lines'])
    address = extract_address(best_result['lines'])
    
    # Fallback cho số CCCD
    if not id_num:
        print("\n⚠️  Thử OCR chuyên dụng cho số...")
        cfg = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        try:
            num_text = pytesseract.image_to_string(img_binary, config=cfg)
            num_text = normalize_to_digits(num_text)
            if 9 <= len(num_text) <= 12:
                id_num = num_text
                print(f"✓ Tìm thấy: {id_num}")
        except:
            pass
    
    # Hiển thị kết quả
    print("\n" + "=" * 80)
    print(" KẾT QUẢ CUỐI CÙNG ".center(80, "="))
    print("=" * 80)
    
    print(f"\n📇 Số CCCD/CMND    : {id_num or '❌ Không tìm thấy'}")
    print(f"👤 Họ và tên       : {name or '❌ Không tìm thấy'}")
    print(f"🎂 Ngày sinh       : {dob or '❌ Không tìm thấy'}")
    print(f"⚧  Giới tính       : {sex or '❌ Không tìm thấy'}")
    print(f"🌍 Quốc tịch       : {nation or '❌ Không tìm thấy'}")
    print(f"🏘  Quê quán        : {origin or '❌ Không tìm thấy'}")
    print(f"🏠 Nơi thường trú  : {address or '❌ Không tìm thấy'}")
    
    # Thống kê
    success_count = sum([1 for x in [id_num, name, dob, sex, nation, origin, address] if x])
    print(f"\n📊 Trích xuất: {success_count}/7 trường")
    
    print("\n" + "=" * 80)
    print(" HOÀN TẤT ".center(80, "="))
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()