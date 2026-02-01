import cv2

def try_open(idx: int):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
    for b in backends:
        cap = cv2.VideoCapture(idx, b) if b != 0 else cv2.VideoCapture(idx)
        opened = cap.isOpened()
        print(f"Index {idx} backend {b} opened: {opened}")
        if opened:
            ret, frame = cap.read()
            print("Read:", ret, "Shape:", None if frame is None else frame.shape)
            cap.release()
            return True
        cap.release()
    return False

if __name__ == "__main__":
    for i in range(6):
        try_open(i)
