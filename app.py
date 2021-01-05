from flask import Flask, render_template, request, send_from_directory
import numpy
from cv2 import cv2
import pytesseract
import csv


app = Flask(__name__)


def pre_processing(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return threshold_img


def parse_text(threshold_img):
    tesseract_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(
        threshold_img, output_type=pytesseract.Output.DICT, config=tesseract_config, lang='eng')
    return details


def format_text(details):
    parse_text = []
    word_list = []
    last_word = ''
    for word in details['text']:
        if word != '':
            word_list.append(word)
            last_word = word
        if (last_word != '' and word == '') or (word == details['text'][-1]):
            parse_text.append(word_list)
            word_list = []
    return parse_text


def write_text(formatted_text):
    with open('resulted_text.txt', 'w', newline="") as file:
        csv.writer(file, delimiter=" ").writerows(formatted_text)
    with open('resulted_text.txt', 'r') as file:
        data = file.read()
    return data


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        img = request.files['img']
        image = cv2.imdecode(numpy.frombuffer(
            img.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        thresholds_image = pre_processing(image)
        parsed_data = parse_text(thresholds_image)
        accuracy_threshold = 30
        arranged_text = format_text(parsed_data)
        data = write_text(arranged_text)
        return render_template('result.html', data=data)
    else:
        return render_template('index.html')


@app.route('/download',  methods=['POST', 'GET'])
def download():
    path = "resulted_text.txt"
    return send_from_directory("", path)


if __name__ == "__main__":
    app.run(debug=True)
