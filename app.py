from flask import Flask, render_template, request, jsonify
import numpy as np
from cv2 import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
import urllib.request
import requests
import io
import PIL
from PIL import Image

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            link = request.form['link']
            response = requests.get(link)
            image_bytes = io.BytesIO(response.content)
            img = PIL.Image.open(image_bytes)
            args = {"east": "./east_text_detection.pb",
                    "min_confidence": 0.5, "width": 320, "height": 320}
            image = np.asarray(img)
            orig = image.copy()
            (origH, origW) = image.shape[:2]
            (newW, newH) = (args["width"], args["height"])
            rW = origW / float(newW)
            rH = origH / float(newH)
            image = cv2.resize(image, (newW, newH))
            (H, W) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
            net = cv2.dnn.readNet(args["east"])
            layerNames = [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"
            ]
            net.setInput(blob)
            (scores, geometry) = net.forward(layerNames)

            def predictions(prob_score, geo):
                (numR, numC) = prob_score.shape[2:4]
                boxes = []
                confidence_val = []
                for y in range(0, numR):
                    scoresData = prob_score[0, 0, y]
                    x0 = geo[0, 0, y]
                    x1 = geo[0, 1, y]
                    x2 = geo[0, 2, y]
                    x3 = geo[0, 3, y]
                    anglesData = geo[0, 4, y]
                    for i in range(0, numC):
                        if scoresData[i] < args["min_confidence"]:
                            continue
                        (offX, offY) = (i * 4.0, y * 4.0)
                        angle = anglesData[i]
                        cos = np.cos(angle)
                        sin = np.sin(angle)
                        h = x0[i] + x2[i]
                        w = x1[i] + x3[i]
                        endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
                        endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
                        startX = int(endX - w)
                        startY = int(endY - h)
                        boxes.append((startX, startY, endX, endY))
                        confidence_val.append(scoresData[i])
                return (boxes, confidence_val)

            (boxes, confidence_val) = predictions(scores, geometry)
            boxes = non_max_suppression(np.array(boxes), probs=confidence_val)
            results = []
            for (startX, startY, endX, endY) in boxes:
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                r = orig[startY:endY, startX:endX]
                configuration = ("-l eng --oem 1 --psm 8")
                text = pytesseract.image_to_string(r, config=configuration)
                results.append(((startX, startY, endX, endY), text))

            orig_image = orig.copy()
            lst = []
            for ((start_X, start_Y, end_X, end_Y), text) in results:
                lst.append(str(text).replace("\n\x0c", ""))
                text = "".join(
                    [x if ord(x) < 128 else "" for x in text]).strip()
                cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
                              (0, 0, 255), 2)
                cv2.putText(orig_image, text, (start_X, start_Y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            lst.reverse()
        except:
            return "Enter the correct jpg image url"
        return jsonify(lst)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
