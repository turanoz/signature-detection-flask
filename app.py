# app.py
from flask import Flask, json, request, jsonify, redirect
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, morphology
from skimage.measure import regionprops
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "tser-coders-devops"
UPLOAD_FOLDER = 'static/uploads'
CONVERTING_FOLDER = 'static/converts'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONVERTING_FOLDER'] = CONVERTING_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(["pdf"])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path, dpi=600)
    for index, image in enumerate(images):
        image.save(f'static/converts/0.jpg')


def show_image(img):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def extract_sign(img, outlier_weight=3, outlier_bias=100, amplfier=10, area_size=10):
    condition = img > img.mean()
    labels = measure.label(condition, background=1)

    total_pixels = 0
    nb_region = 0
    average = 0.0
    for region in regionprops(labels):
        if (region.area > area_size):
            total_pixels += region.area
            nb_region += 1

    average = (total_pixels / nb_region)
    small_size_outlier = average * outlier_weight + outlier_bias
    big_size_outlier = small_size_outlier * amplfier
    pre_version = morphology.remove_small_objects(labels, small_size_outlier)
    component_sizes = np.bincount(pre_version.ravel())
    too_small = component_sizes > (big_size_outlier)
    too_small_mask = too_small[pre_version]
    pre_version[too_small_mask] = 0
    labeled_mask = np.full(pre_version.shape, 255, dtype="uint8")
    labeled_mask = labeled_mask * (pre_version == 0)

    return labeled_mask


def is_intersected(box_a, box_b) -> bool:
    [x_a, y_a, w_a, h_a] = box_a
    [x_b, y_b, w_b, h_b] = box_b
    if y_a > y_b + h_b: return False
    if y_a + h_a < y_b: return False
    if x_a > x_b + w_b: return False
    if x_a + w_a < x_b: return False
    return True


def merge_boxes(box_a, box_b) -> list:
    [x_a, y_a, w_a, h_a] = box_a
    [x_b, y_b, w_b, h_b] = box_b
    min_x = min(x_a, x_b)
    min_y = min(y_a, y_b)
    max_w = max(w_a, w_b, (x_b + w_b - x_a), (x_a + w_a - x_b))
    max_h = max(h_a, h_b, (y_b + h_b - y_a), (y_a + h_a - y_b))

    return [min_x, min_y, max_w, max_h]


@app.route('/')
def main():
    return '<h1>YETKİSİZ ERİŞİM</h1>'


@app.route('/signature-detection', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        resp = jsonify({'message': 'Dosya göndermediniz'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    errors = {}
    success = False
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors[file.filename] = 'İzin verilmeyen dosya uzantısı'
    if success and errors:
        errors['message'] = 'Dosya Yüklendi'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        convert_pdf_to_images(f'{UPLOAD_FOLDER}/{filename}')
        os.remove("static/uploads/" + filename)
        # Signature Detection :
        #https://github.com/EnzoSeason/signature_detection opencv methodları dahil...
        image = cv2.imread('static/converts/0.jpg')
        frame_HSV = cv2.cvtColor(image[0:2500, 1000:2000], cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (0, 0, 250), (255, 255, 255))
        preview = extract_sign(frame_threshold)
        cnts = cv2.findContours(preview, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = cnts[0] if len(cnts) == 2 else cnts[1]
        boxes = []
        copy_img = preview.copy()
        for c in cnt:
            (x, y, w, h) = cv2.boundingRect(c)
            if h * w > 10000 and h < copy_img.shape[0] and w < copy_img.shape[
                1] and x < 350 and y < 1720 and x != 0 and h > 83:
                cv2.rectangle(copy_img, (x, y), (x + w, y + h), (155, 155, 0), 1)
                boxes.append([x, y, w, h])
        np_boxes = np.array(boxes)
        area_size = list(map(lambda box: box[2] * box[3], np_boxes))
        area_size = np.array(area_size)
        area_dec_order = area_size.argsort()[::-1]
        sorted_boxes = np_boxes[area_dec_order]
        regions = {}
        for i, box in enumerate(sorted_boxes):
            if len(regions) == 0:
                regions[0] = box
            else:
                is_merged = False
                for key, region in regions.items():
                    if is_intersected(region, box) == True:
                        new_region = merge_boxes(region, box)
                        regions[key] = new_region
                        is_merged = True
                        break
                if is_merged == False:
                    key = len(regions)
                    regions[key] = box
        os.remove("static/converts/0.jpg")
        if regions:
            return jsonify(imza="true"), 200
        else:
            return jsonify(imza="false"), 200
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp


if __name__ == '__main__':
    app.run()
