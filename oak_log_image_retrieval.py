from PIL import Image
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import sys
import numpy as np
import cv2
import pickle

################## phase 1 ####################

target_shape = (200, 200)
path = ''
model = load_model(path + 'embedding_256_2813_30_100.h5')

def image_to_embedding(image_path, model):
    img = Image.open(image_path)
    img = img.resize(target_shape)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    embedding_vector = model.predict(img)[0]
    #embedding_vector /= np.linalg.norm(embedding_vector)
    return embedding_vector

log_embeddings = {}
# Load image_embeddings from the file
with open(path+'image_embeddings_256_2831_30_100.pkl', 'rb') as f:
    log_embeddings = pickle.load(f)

def image_retrieval(query_image_path):
    query_embedding = image_to_embedding(query_image_path, model)
    similar_images = []
    for image_path, embedding in log_embeddings.items():
        distance = np.linalg.norm(embedding - query_embedding)
        similar_images.append((image_path, distance))

    similar_images.sort(key=lambda x: x[1])

    limited_distance = similar_images[0][1] / 0.6

    good_similar_images =[]

    for i in range(len(similar_images)):
        if similar_images[i][1] <= limited_distance:
            good_similar_images.append(similar_images[i][0])
        else:
            break

    return good_similar_images

################## phase 2 ####################

class Element:
    def __init__(self, filename):
        self.image = None
        self.keypoints = []
        self.descriptors = []
        self.label = ''
        self.name = filename
        self.file_data_name = os.path.splitext(self.name)[0] + '.pkl'
    def keypoint_extraction(self):
        img = cv2.imread(self.name, cv2.IMREAD_GRAYSCALE)
        self.image = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        detector = cv2.SIFT_create()
        self.keypoints, self.descriptors = detector.detectAndCompute(self.image, None)

    def read_keypoint_extraction(self):
        # Check if keypoints and descriptors exist in the saved file
        if os.path.exists(self.file_data_name):
            with open(self.file_data_name, 'rb') as file:
                data = pickle.load(file)
                self.image = cv2.cvtColor(
                    cv2.resize(
                        np.array(data['image'], dtype=np.uint8),
                        None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR
                    ),
                    cv2.COLOR_GRAY2BGR
                )
                self.keypoints = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in data['keypoints']]
                self.descriptors = np.array(data['descriptors'], dtype=np.float32)
                self.label = data['label']
        else:
            img = cv2.imread(self.name, cv2.IMREAD_GRAYSCALE)
            self.image = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            detector = cv2.SIFT_create()
            self.keypoints, self.descriptors = detector.detectAndCompute(self.image, None)

            # Save the extracted keypoints and descriptors
            self.save()

    def save(self):
        data = {
            'image': self.image.tolist(),
            'keypoints': [kp.pt for kp in self.keypoints],
            'descriptors': self.descriptors.tolist(),
            'label': self.label,
            'name': self.name
        }
        with open(self.file_data_name, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            element = Element(data['name'])
            element.image = cv2.cvtColor(
                cv2.resize(
                    np.array(data['image'], dtype=np.uint8),
                    None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR
                ),
                cv2.COLOR_GRAY2BGR
            )
            element.keypoints = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in data['keypoints']]
            element.descriptors = np.array(data['descriptors'], dtype=np.float32)
            element.label = data['label']
            return element

def read_logs(file_list):
    # Read logs
    #print(nb_logs)
    for log_name in file_list:
        #print(log_name)
        e = Element(str(log_name))
        e.read_keypoint_extraction()
        logs.append(e)
        #print(log_name,'->',len(logs[len(logs)-1].keypoints))

def read_image(image_name):
    e = Element(image_name)
    e.keypoint_extraction()
    return e

def is_nearly_rectangle(corners, angle_threshold):
    # Ensure we have exactly four corners
    if len(corners) != 4:
        return False

    point1, point2, point3, point4 = corners

    # Calculate the angle for each edge
    edge1 = point2 - point1
    edge2 = point3 - point2
    edge3 = point4 - point3
    edge4 = point1 - point4

    angle1 = np.degrees(np.arccos(np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2))))
    angle2 = np.degrees(np.arccos(np.dot(edge2, edge3) / (np.linalg.norm(edge2) * np.linalg.norm(edge3))))
    angle3 = np.degrees(np.arccos(np.dot(edge3, edge4) / (np.linalg.norm(edge3) * np.linalg.norm(edge4))))
    angle4 = np.degrees(np.arccos(np.dot(edge4, edge1) / (np.linalg.norm(edge4) * np.linalg.norm(edge1))))

    if any(abs(angle - 90.0) > angle_threshold for angle in [angle1, angle2, angle3, angle4]):
        return False

    return True

def matching(log, image):
    # Matching keypoints
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(log.descriptors, image.descriptors, 2)

    # Filtering matches using ratio
    good_matches = []
    ratio_thresh = 0.75
    for match1, match2 in knn_matches:
        if match1.distance < ratio_thresh * match2.distance:
            good_matches.append(match1)
    if len(good_matches) < 4:
        return 0.0, None, None

    # Estimating a bounding box of the object
    kp_log = [log.keypoints[match.queryIdx].pt for match in good_matches]
    kp_image = [image.keypoints[match.trainIdx].pt for match in good_matches]
    H, inlierMask = cv2.findHomography(np.float32(kp_log), np.float32(kp_image), cv2.RANSAC, 3)

    # Get the corners from the log image
    log_corners = np.array([[0, 0], [log.image.shape[1], 0], [log.image.shape[1], log.image.shape[0]], [0, log.image.shape[0]]], dtype=np.float32)
    image_corners = cv2.perspectiveTransform(log_corners.reshape(-1, 1, 2), H).reshape(-1, 2)

    # Checking the bounding box is a convex hull
    if not cv2.isContourConvex(image_corners):
        return 0.0, None, None

    # Checking the bounding box is a nearly rectangle
    if not is_nearly_rectangle(image_corners, 20):
        return 0.0, None, None

    # Filtering inlier matches
    inlier_matches = []
    for i in range(inlierMask.shape[0]):
        if inlierMask[i]:
            inlier_matches.append(good_matches[i])

    return len(inlier_matches) / len(log.descriptors), inlier_matches, H

def compare_label(path1, path2):
    filename1 = os.path.basename(path1)
    filename2 = os.path.basename(path2)
    filename1 = filename1.rsplit('_',1)[0]
    filename2 = filename2.rsplit('_',1)[0]
    return filename1 == filename2

query_image_path = sys.argv[1]

logs = []
image_retrieval_list = image_retrieval(query_image_path)
read_logs(image_retrieval_list)
image = read_image(query_image_path)

found = False
best_rs = -1
p = -1
for l in range(len(logs)):
    r, good_matches, H = matching(logs[l], image)
    if r >= 0.0005:
        if r > best_rs:
            best_rs = r
            p = l
        found = True

if found:
    print("The query image " + query_image_path + " match with " + logs[p].name)
    if compare_label(query_image_path,logs[p].name):
        print('=> CORRECT MATCHING')
    else:
        print('=> incorrect matching')
else:
    print("The query image " + query_image_path + " does not match with any log.")

print("Finished")
