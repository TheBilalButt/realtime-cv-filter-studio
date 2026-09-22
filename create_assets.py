import cv2
import numpy as np
import os

def create_sample_images(output_dir="assets"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create a colorful portrait-style demo image (with face/person silhouette, vibrant colors)
    h, w = 600, 800
    portrait = np.zeros((h, w, 3), dtype=np.uint8)
    # Background gradient
    for y in range(h):
        r = int(70 + 100 * (y / h))
        g = int(120 + 80 * (1 - y / h))
        b = int(180 + 70 * (y / h))
        portrait[y, :] = [b, g, r]
        
    # Draw a stylized person figure in the foreground
    # Head
    cv2.circle(portrait, (400, 240), 90, (180, 200, 230), -1)
    # Hair
    cv2.ellipse(portrait, (400, 200), (95, 80), 0, 160, 380, (40, 30, 25), -1)
    # Eyes
    cv2.circle(portrait, (365, 235), 10, (50, 40, 30), -1)
    cv2.circle(portrait, (435, 235), 10, (50, 40, 30), -1)
    cv2.circle(portrait, (367, 233), 3, (255, 255, 255), -1)
    cv2.circle(portrait, (437, 233), 3, (255, 255, 255), -1)
    # Smile
    cv2.ellipse(portrait, (400, 275), (35, 20), 0, 10, 170, (80, 80, 200), 4)
    # Shoulders / Torso
    pts = np.array([[220, 600], [300, 360], [500, 360], [580, 600]], np.int32)
    cv2.fillPoly(portrait, [pts], (220, 110, 40))
    # Shirt collar
    cv2.fillConvexPoly(portrait, np.array([[360, 360], [400, 420], [440, 360]]), (245, 245, 250))
    
    # Add some foreground details
    cv2.putText(portrait, "Filter Studio Demo", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imwrite(os.path.join(output_dir, "portrait.jpg"), portrait)
    
    # 2. Create landscape demo image with geometric shapes and fine textures
    landscape = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky gradient
    for y in range(350):
        b = int(240 - y * 0.3)
        g = int(180 - y * 0.2)
        r = int(120 - y * 0.2)
        landscape[y, :] = [b, g, r]
    # Mountains
    pts1 = np.array([[0, 400], [200, 150], [450, 380], [700, 180], [800, 320], [800, 600], [0, 600]], np.int32)
    cv2.fillPoly(landscape, [pts1], (100, 90, 80))
    pts2 = np.array([[100, 420], [320, 230], [520, 410], [800, 430], [800, 600], [0, 600]], np.int32)
    cv2.fillPoly(landscape, [pts2], (60, 120, 70))
    # Sun
    cv2.circle(landscape, (680, 120), 45, (120, 220, 255), -1)
    
    cv2.imwrite(os.path.join(output_dir, "landscape.jpg"), landscape)

    # 3. Create realistic sample document page with uneven shadow and handwritten notes
    doc = np.full((h, w, 3), 205, dtype=np.uint8)
    for y in range(h):
        shadow = int(65 * (y / h))
        doc[y, :] = np.clip(doc[y, :] - shadow, 0, 255)
    
    cv2.putText(doc, "LECTURE NOTES - COMPUTER VISION", (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.9, (10, 10, 120), 2)
    cv2.line(doc, (40, 85), (760, 85), (80, 80, 80), 2)
    cv2.putText(doc, "1. Real-time paper and document cleaning", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2)
    cv2.putText(doc, "   - Normalizes background illumination across shadows", (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
    cv2.putText(doc, "   - Eliminates yellow tint, creases and low-light noise", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
    cv2.putText(doc, "   - Turns paper surface into pure 255 white", (50, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
    cv2.putText(doc, "2. Handwritten pen ink preserved with high contrast", (50, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 20, 20), 2)
    cv2.putText(doc, "   - Blue ink, red pen, and graphite marks stay crisp", (50, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 160), 2)
    cv2.putText(doc, "Sign: Bilal Butt | Verified Clean Page", (50, 480), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.1, (10, 10, 10), 2)
    
    cv2.imwrite(os.path.join(output_dir, "document.jpg"), doc)
    print("Created sample images successfully.")

if __name__ == "__main__":
    create_sample_images()

