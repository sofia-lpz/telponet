import cv2
from ultralytics import YOLO, solutions

# Load your trained YOLO model
model = YOLO("drone.pt")

# Open the default camera (device index 0), change index if needed (e.g., 1 for an external camera)
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error opening webcam"

# Get camera properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Init heatmap object
heatmap_obj = solutions.Heatmap(
    colormap=cv2.COLORMAP_PARULA,
    view_img=True,
    shape="square",
    names=model.names,
)

# Continuously read frames from the live feed
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Error reading frame from camera.")
        break

    # Perform tracking using YOLO model
    tracks = model.track(im0, persist=True, show=False)

    # Generate and overlay heatmap on the frame
    im0 = heatmap_obj.generate_heatmap(im0, tracks)

    # Display the frame with heatmap in a window
    cv2.imshow("Live Feed with Heatmap", im0)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
