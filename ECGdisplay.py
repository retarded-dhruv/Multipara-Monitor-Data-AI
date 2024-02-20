import cv2
import numpy as np
import pytesseract

# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    return edged

# Function to find contours and detect text regions
def detect_text_regions(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if aspect_ratio > 2 and area > 1000:
            text_regions.append((x, y, x + w, y + h))
    return text_regions

# Function to extract text from regions
def extract_text(image, regions):
    extracted_text = []
    for region in regions:
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(roi, config='--psm 6')
        extracted_text.append((text, (x1, y1, x2, y2)))
    return extracted_text

# Function to filter text based on keywords
def filter_text(text_regions, keywords):
    filtered_regions = []
    for text, region in text_regions:
        for keyword in keywords:
            if keyword.lower() in text.lower():
                filtered_regions.append(region)
                break
    return filtered_regions

# Function to draw rectangles around detected regions
def draw_regions(image, regions):
    for region in regions:
        x1, y1, x2, y2 = region
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Main function
def main():
    video_capture = cv2.VideoCapture(0)  # You can replace 0 with the video file path
    keywords = ['HR', 'ECG', 'Oxygen']  # Keywords to identify critical data areas
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        processed_frame = preprocess_image(frame)
        text_regions = detect_text_regions(processed_frame)
        filtered_regions = filter_text(extract_text(frame, text_regions), keywords)
        draw_regions(frame, filtered_regions)
        cv2.imshow('Monitor Data Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
