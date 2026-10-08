import cv2
import torch
from PIL import Image

def is_raspberry_pi():
    try:
        with open('/sys/firmware/devicetree/base/model', 'r') as f:
            return 'raspberry pi' in f.read().lower()
    except:
        return False

# Camera inference loop
def segment(image, background, threshold=25):
    # Compute absolute difference between background and current frame
    diff = cv2.absdiff(background, image)
    # Threshold to get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    # Find contours
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None
    else:
        # Return the largest contour (the hand)
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def run_camera(model, class_names, device, transform,
               cam_index: int = 3,
               width: int = 640,
               height: int = 480) -> None:
    # ROI Coordinates (top, right, bottom, left)
    top, right, bottom, left = 10, 350, 225, 590

    # Normally, 0 for built-in webcam and 1&2 for external USB devices
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    bg = None

    print("Press 's' to set background, 'q' to quit.")

    model.eval()

    raspberry_pi = False
    if is_raspberry_pi():
        from utils.init_hand import create_hand
        hand, gestures = create_hand()
        raspberry_pi = True


    last_pred = ""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        # Clone the frame so we can draw the box without affecting processing
        clone = frame.copy()
    
        # Extract the ROI from the frame
        roi = frame[top:bottom, right:left]
    
        # Pre-process the ROI
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord('s'):
            bg = gray_roi.copy().astype("uint8")
            print("Background ROI captured!")

        if bg is not None:
            hand_data = segment(gray_roi, bg)
        
            if hand_data is not None:
                (thresholded, segmented) = hand_data
                # Draw the contour (offset by ROI coordinates to align with main frame)
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(segmented)
                hand_crop = roi[y:y + h, x:x + w]
                if hand_crop.size != 0:
                    hand_resized = cv2.resize(hand_crop, (32, 32))
                    hand_rgb = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2RGB)
                    hand_pil = Image.fromarray(hand_rgb)

                    with torch.no_grad():
                        inp = transform(hand_pil).unsqueeze(0).to(device)

                        logits = model(inp)
                        probs = torch.softmax(logits, dim=1)
                        pred_id = logits.argmax(dim=1).item()
                        pred_name = class_names[pred_id]
                        confidence = probs[0, pred_id].item()

                    cv2.putText(clone,
                                f'{pred_name}: {confidence:.2f}',
                                (right + 10, top + 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2,
                                cv2.LINE_AA)

                    if pred_name != last_pred:
                        print(f'Prediction: {pred_name}')
                        last_pred = pred_name
                    if raspberry_pi:
                        # Game Logic
                        match pred_name:
                            case 'Rock':
                                hand.paper()
                            case 'Paper':
                                hand.scissors()
                            case 'Scissors':
                                hand.rock()

                cv2.imshow("Thresholded ROI", thresholded)

        # Draw the bounding box on the main display
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow('Video Feed', clone)

        if keypress == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if raspberry_pi:
        hand.stop()
