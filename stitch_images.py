import cv2, urllib.request
import numpy as np


# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):
    # Get dimensions of input images
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Get the canvas dimensions
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # Get perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims, M)

    # Result dimensions
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))

    result_img[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[0]] = img1

    return result_img


# Find SIFT and return Homography Matrix
def get_homography(img1, img2):
    # Equalize histogram
    img1_h = equalize_histogram(img1)
    img2_h = equalize_histogram(img2)

    #Sharpen images
    img1_sharp = sharpen_image(img1_h)
    img2_sharp = sharpen_image(img2_h)

    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    # Get keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1_sharp, None)
    kp2, des2 = sift.detectAndCompute(img2_sharp, None)

    # Bruteforce matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Check against Lowe ratio test
    verify_ratio = 0.85
    verified_matches = []
    for m1, m2 in matches:
        if m1.distance < verify_ratio * m2.distance:
            verified_matches.append(m1)

    # Mimnum number of matches
    min_matches = 8
    if len(verified_matches) > min_matches:

        # Array to store matching points
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for match in verified_matches:
            img1_pts.append(kp1[match.queryIdx].pt)
            img2_pts.append(kp2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        # Compute homography matrix
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img_matched = cv2.drawMatches(img1, kp1, img2, kp2, verified_matches, None, **draw_params)
        
        return M, img_matched
    else:
        print('Error: Not enough matches')
        exit()

# Sharpen Images
def sharpen_image(img):
    # img1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # img1_sharp = cv2.filter2D(img1_gray, -1, kernel)

    img_hsv[:, :, 1] = cv2.filter2D(img_hsv[:, :, 1], -1, kernel)

    # img_sharp = np.concatenate((img1, cv2.cvtColor(img1_hsv, cv2.COLOR_HSV2BGR)), axis=1)

    img_sharp = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)

    return img_sharp


# Equalize Histogram of Images
def equalize_histogram(img):
    # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    # img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


    return img


def get_image_by_url(img1_url, img2_url):
    img1 = urllib.request.urlopen(img1_url)
    img2 = urllib.request.urlopen(img2_url)
    img1 = np.asarray(bytearray(img1.read()), dtype=np.uint8)
    img2 = np.asarray(bytearray(img2.read()), dtype=np.uint8)
    img1 = cv2.imdecode(img1, -1)
    img2 = cv2.imdecode(img2, -1)
    get_image(img1, img2)
    

def get_default_images(image_code):
    if image_code == 'a':
        img1 = cv2.imread(r'images/bldg1.jpg')
        img2 = cv2.imread(r'images/bldg2.jpg')
    elif image_code == 'b':
        img1 = cv2.imread(r'images/coast1.jpg')
        img2 = cv2.imread(r'images/coast2.jpg')
    elif image_code == 'c':
        img1 = cv2.imread(r'images/room1.jpg')
        img2 = cv2.imread(r'images/room2.jpg')
    elif image_code == 'd':
        img1 = cv2.imread(r'images/tower1.jpg')
        img2 = cv2.imread(r'images/tower2.jpg')
    else:
        img1 = cv2.imread(r'images/bldg1.jpg')
        img2 = cv2.imread(r'images/bldg2.jpg')
    get_image(img1, img2)


# Main function definition
def get_image(img1, img2):

    # Get homography matrix and matched image with keypoints
    M, img_matched = get_homography(img1, img2)

    # Stitch the images together
    result_image = get_stitched_image(img2, img1, M)
    # result_image = get_stitched_image(equalize_histogram(img2), equalize_histogram(img1), M)

    # Write the result to /static directory
    cv2.imwrite('static/result_image.jpg', result_image)
    cv2.imwrite('static/img1.jpg',img1)
    cv2.imwrite('static/img2.jpg',img2)
    cv2.imwrite('static/img_matched.jpg',img_matched)


# Call main function
if __name__ == '__main__':
    # get_image('a')
    # get_image_by_url('https://raw.githubusercontent.com/pavanpn/Image-Stitching/master/images/04_cornerA.jpg',
    #                  'https://raw.githubusercontent.com/pavanpn/Image-Stitching/master/images/04_cornerB.jpg')
    get_default_images('d')
