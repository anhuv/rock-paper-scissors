from keras.models import load_model
import cv2
import numpy as np
from random import choice

# Sử dụng mạng neutral nhận biết cử chỉ tay của người chơi qua webcame máy tính
# ramdom.choice là lựa chọn của máy tính
# nhấn phím q để quit 

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Hoa"

    if move1 == "rock":
        if move2 == "scissors":
            return "Nguoi choi"
        if move2 == "paper":
            return "May"

    if move1 == "paper":
        if move2 == "rock":
            return "Nguoi choi"
        if move2 == "scissors":
            return "May"

    if move1 == "scissors":
        if move2 == "paper":
            return "Nguoi choi"
        if move2 == "rock":
            return "May"

#load model
model = load_model("rock-paper-scissors-model.h5")

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

prev_move = None

while True:    

    ret, frame = cap.read()
    if not ret:
        continue

    # vùng hiển thị của người chơi
    cv2.rectangle(frame, (100, 100), (500, 500), (0, 255, 0), 2)
    # vùng hiển thị của máy
    cv2.rectangle(frame, (800, 100), (1200, 500), (0, 255, 0), 2)

    # trích xuất ảnh trong vùng hiển thị của người chơi, resize theo đầu vào của model đã train
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # dự đoán ảnh sử dụng model
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    # Hiển thị thông tin
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Nguoi choi chon: " + user_move_name,
                (50, 50), font, 1.2, (204, 0, 204), 2, cv2.LINE_AA)
    cv2.putText(frame, "May chon: " + computer_move_name,
                (750, 50), font, 1.2, (204, 0, 204), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, (255, 0, 0), 4, cv2.LINE_AA)

    if computer_move_name != "none":
        icon = cv2.imread(
            "images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (400, 400))
        frame[100:500, 800:1200] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
