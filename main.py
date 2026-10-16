def get_color_name_hsv(hsv):
    h, s, v = hsv
    if v < 50:
        return 'Black'
    if v > 200 and s < 40:
        return 'White'
    if s < 40:
        return 'Gray'
    if h < 10 or h >= 170:
        return 'Red'
    if 10 <= h < 25:
        return 'Orange'
    if 25 <= h < 35:
        return 'Yellow'
    if 35 <= h < 85:
        return 'Green'
    if 85 <= h < 125:
        return 'Cyan'
    if 125 <= h < 145:
        return 'Blue'
    if 145 <= h < 170:
        return 'Magenta'
    return 'Unknown'
import cv2
import numpy as np

def get_dominant_color(image, k=3):
    # Reshape image to a list of pixels
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Count labels to find most popular
    _, counts = np.unique(labels, return_counts=True)
    dominant = centers[np.argmax(counts)]
    return dominant.astype(int)

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally to avoid mirror image
        frame = cv2.flip(frame, 1)

        # Get center region
        h, w, _ = frame.shape
        size = 50
        x1, y1 = w//2 - size, h//2 - size
        x2, y2 = w//2 + size, h//2 + size
        roi = frame[y1:y2, x1:x2]

        # Get dominant color (BGR)
        dominant = get_dominant_color(roi, k=3)
        # Convert to HSV
        dominant_bgr = np.uint8([[dominant]])
        dominant_hsv = cv2.cvtColor(dominant_bgr, cv2.COLOR_BGR2HSV)[0][0]
        color_name = get_color_name_hsv(dominant_hsv)
        color_text = f'Color: {color_name} | HSV: {dominant_hsv[0]}, {dominant_hsv[1]}, {dominant_hsv[2]}'

        # Draw rectangle and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, color_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        cv2.imshow('KMeans Color Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import numpy as np

# Example training data: [B, G, R] values and their color labels
samples = np.array([
    [255, 0, 0],    # Blue
    [0, 255, 0],    # Green
    [0, 0, 255],    # Red
    [0, 255, 255],  # Yellow
    [255, 255, 0],  # Cyan
    [255, 0, 255],  # Magenta
], dtype=np.float32)

labels = np.array([0, 1, 2, 3, 4, 5])  # 0: Blue, 1: Green, etc.

# Create and train KNN
knn = cv2.ml.KNearest_create()
knn.train(samples, cv2.ml.ROW_SAMPLE, labels)

color_names = ['Blue', 'Green', 'Red', 'Yellow', 'Cyan', 'Magenta']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the center pixel
    h, w, _ = frame.shape
    center = frame[h//2, w//2]
    sample = np.array([center], dtype=np.float32)

    # Predict color
    ret, result, neighbours, dist = knn.findNearest(sample, k=3)
    color_idx = int(result[0][0])
    color_label = color_names[color_idx]

    # Draw result
    cv2.putText(frame, f'Color: {color_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.circle(frame, (w//2, h//2), 5, (0,0,0), 2)

    cv2.imshow('Color Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()