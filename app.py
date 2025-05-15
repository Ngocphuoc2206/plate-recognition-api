from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import easyocr

app = FastAPI()
reader = easyocr.Reader(['en'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_text(text):
    text = text.upper().strip()
    text = text.replace(" ", "")
    text = text.replace("O", "0").replace("I", "1").replace("B", "8").replace("S", "5")
    text = ''.join(c for c in text if c.isalnum() or c in "-")
    return text

@app.post("/recognize-plate/")
async def recognize_plate(file: UploadFile = File(...)):
    try:
        # Đọc file ảnh
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Nếu ảnh không hợp lệ
        if img is None:
            raise HTTPException(status_code=400, detail="Không thể đọc ảnh")

        # Resize ảnh
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Áp dụng thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Dùng EasyOCR để nhận diện văn bản
        results = reader.readtext(thresh)

        valid_lines = []
        for (bbox, text, conf) in results:
            cleaned = clean_text(text)
            if len(cleaned) >= 2 and conf > 0.3:  # giữ lại text >=2 ký tự và độ tin cậy > 0.3
                y_center = (bbox[0][1] + bbox[2][1]) / 2
                valid_lines.append((y_center, cleaned, conf))  # Thêm confidence vào tuple

        valid_lines.sort(key=lambda x: x[0])  # sắp xếp từ trên xuống dưới theo tọa độ Y

        plate_text = ''
        confidence = 0.0
        if len(valid_lines) >= 2:
            # Nếu có ít nhất 2 dòng, giả sử biển số vuông
            line1, text1, conf1 = valid_lines[0]
            line2, text2, conf2 = valid_lines[1]
            plate_text = f"{text1}{text2}"
            confidence = (conf1 + conf2) / 2 if valid_lines else 0.0  # Tính trung bình confidence
        elif len(valid_lines) == 1:
            # Nếu chỉ có 1 dòng (biển dài)
            plate_text = valid_lines[0][1]
            confidence = valid_lines[0][2]
        else:
            plate_text = "Không nhận diện được"
            confidence = 0.0

        return {
            "plate": plate_text,
            "confidence": round(confidence * 100, 2),  # Trả về độ tin cậy đã làm tròn
            "all": [{"text": line[1], "confidence": round(line[2] * 100, 2)} for line in valid_lines]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
