import cv2
import time


def detect_abandoned_item(video_file):
    video = cv2.VideoCapture(video_file)
    ret, frame = video.read()

    dict_of_coord = {}

    color = (0, 0, 255)
    txt = 'ATTENTION! UNKNOWN OBJECT'

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.MP4', fourcc, 20.0, (1280, 960))  # record video in a separate mp4 file

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    while True:
        time.sleep(0.03)  # video speed adjustment

        ret, frame = video.read()
        gray_frame_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        frame_diff = cv2.absdiff(gray_frame, gray_frame_next)  # calculating the pixel difference between frames
        _, thresh_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)  # applying the difference threshold

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # noise removal
        thresh = cv2.morphologyEx(thresh_diff, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # finding contours

        for contour in contours:
            area = cv2.contourArea(contour)  # the area inside the contour
            x, y, w, h = cv2.boundingRect(contour)

            if area > 3000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # drawing contours
                coordinates = (x, y, x + w, y + h)

                if coordinates in dict_of_coord:
                    dict_of_coord[coordinates] += 1
                else:
                    dict_of_coord[coordinates] = 1

                if dict_of_coord[coordinates] > 100:  # the coordinates of the object remain unchanged for more than 100 frames
                    cv2.putText(frame, "abandoned item!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  
                    cv2.putText(frame, txt, (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)  # inserting an alarm in the upper left corner
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Abandoned Items Detection", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

