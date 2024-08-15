import cv2
import os

# Set up video capture
video_path = r'C:\Users\Akansha Dhami\Desktop\something\tere_bina.mp4'   # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create directory to save frames
output_dir = r'C:\Users\Akansha Dha mi\Desktop\something\img'
os.makedirs(output_dir, exist_ok=True)

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 0.5)  # Interval in frames (0.5 seconds)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Save frames at the specified interval
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_interval == 0:
        frame_filename = os.path.join(output_dir, f'frame_{count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        count += 1

# Release video capture
cap.release()
cv2.destroyAllWindows()
