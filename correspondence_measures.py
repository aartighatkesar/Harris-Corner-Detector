import numpy as np
from detect_harris_corners import *


def ssd_correspondence(img_1, img_2, corners_xy_1, corners_xy_2, threshold=0, kernel_size=21):

    # Sum of squared difference

    # this is list of lists [[ind_1, ind_2, ssd_score], []]
    #  where ind_2 is index in 2nd img corner points matched with 1st corner point
    corr_1_to_2 = []

    win_pixels_2 = get_window(corners_xy_2, img_2, kernel_size)

    # Padding for corners
    pad = int(kernel_size/2)
    img_1 = np.pad(img_1, ((pad, pad), (pad, pad)), mode='edge')

    for i, corner_xy in enumerate(corners_xy_1 + pad):
        win_pixels_1 = img_1[corner_xy[1] - pad:corner_xy[1] + pad + 1, corner_xy[0] - pad:corner_xy[0] + pad + 1].flatten()

        ssd_score = np.sum((win_pixels_1 - win_pixels_2)**2, axis=1)
        ind_ssd = np.argmax(ssd_score)

        corr_1_to_2.append([i, ind_ssd, ssd_score[ind_ssd]])

    corr_1_to_2 = sorted(corr_1_to_2, key=lambda x: x[2])

    return corr_1_to_2


def ncc_correspondence(img_1, img_2, corners_xy_1, corners_xy_2, threshold, kernel_size=21):
    # Normalized cross correlation

    # this is list of lists [[ind_1, ind_2, ssd_score], []]
    #  where ind_2 is index in 2nd img corner points matched with 1st corner point
    corr_1_to_2 = []

    win_pixels_2 = get_window(corners_xy_2, img_2, kernel_size)
    win_pixels_2 = win_pixels_2 - np.mean(win_pixels_2, axis=1, keepdims=True)

    # Padding for corners
    pad = int(kernel_size / 2)
    img_1 = np.pad(img_1, ((pad, pad), (pad, pad)), mode='edge')

    for i, corner_xy in enumerate(corners_xy_1 + pad):
        win_pixels_1 = img_1[corner_xy[1] - pad:corner_xy[1] + pad + 1,
                       corner_xy[0] - pad:corner_xy[0] + pad + 1].flatten()

        win_pixels_1 = win_pixels_1 - np.mean(win_pixels_1)

        den = np.sqrt(np.sum(win_pixels_1**2) * np.sum(win_pixels_2**2, axis=1))

        ncc_score= np.sum((win_pixels_1 * win_pixels_2),axis=1)/ den

        ind_ssd = np.argmax(ncc_score)

        corr_1_to_2.append([i, ind_ssd, ncc_score[ind_ssd]])

    corr_1_to_2 = sorted(corr_1_to_2, key=lambda x: x[2], reverse=True)

    return corr_1_to_2


def euclidean_correspondence():
    # Minimum euclidean distance between feature descriptor vectors
    pass


def get_window(xy_crd, img_gray, kernel_size):

    win = int(kernel_size / 2)
    # Pad image to address corners near boundaries of image
    img_pad =  np.pad(img_gray, ((win, win), (win, win)), mode='edge')

    # the location of corners moves due to padding
    xy_crd = xy_crd + win
    ind = xy_crd[:, (1,0)] # converting to (r, c) notation

    def apply_func(ind_row, img, win):
        # print(ind_row)
        out = img[np.ix_(np.arange(ind_row[0]-win,ind_row[0]+win+1),
                              np.arange(ind_row[1]-win,ind_row[1]+win+1))].flatten()
        return out


    pix_win = np.apply_along_axis(apply_func, 1, ind, img_pad, win)

    return pix_win


def draw_correspondence(img_1, img_2, corr_1_to_2, pts_xy_1, pts_xy_2, dest_img_path='', max_lines=100):

    h, w, _ = img_1.shape

    # for i in range(pts_xy_1.shape[0]):
    #     cv2.circle(img_1, (pts_xy_1[i, 0], pts_xy_1[i, 1]), radius=3, color=[0, 255, 0], thickness=-1)
    #
    # for i in range(pts_xy_2.shape[0]):
    #     cv2.circle(img_2, (pts_xy_2[i, 0], pts_xy_2[i, 1]), radius=3, color=[0, 255, 0], thickness=-1)

    img_combined = np.hstack((img_1, img_2))

    for i in range(max_lines):
        x1, y1 = pts_xy_1[corr_1_to_2[i][0]]
        x2, y2 = pts_xy_2[corr_1_to_2[i][1]]
        x2 = x2 + w

        cv2.circle(img_combined, (x1, y1), radius=3, color=[0, 255, 0], thickness=-1)
        cv2.circle(img_combined, (x2, y2), radius=3, color=[0, 255, 0], thickness=-1)
        cv2.line(img_combined, (x1, y1), (x2, y2), (255, 0, 0), 1)

    if dest_img_path:
        cv2.imwrite(dest_img_path, img_combined)


def run_main(img_path, dest_img_path, nms_kernel_size, sigma=2, k=0.06, thresh=0.75, max_no_corners=-1, mode="ssd", corr_kernel_size=21):

    corners_xy_1 = detect_corners(img_path[0], dest_img_path='', nms_kernel_size=nms_kernel_size,
                                  sigma=sigma, k=k, thresh=thresh, max_no_corners=max_no_corners)

    corners_xy_2 = detect_corners(img_path[1], dest_img_path='', nms_kernel_size=nms_kernel_size,
                                  sigma=sigma, k=k, thresh=thresh, max_no_corners=max_no_corners)

    img_bgr_1 = load_image(img_path[0])
    img_1 = cv2.cvtColor(img_bgr_1, cv2.COLOR_BGR2GRAY)
    img_bgr_2 = load_image(img_path[1])
    img_2 = cv2.cvtColor(img_bgr_2, cv2.COLOR_BGR2GRAY)


    if mode == 'ssd':
        corr_1_to_2 = ssd_correspondence(img_1, img_2, corners_xy_1, corners_xy_2, threshold=0, kernel_size=corr_kernel_size)
    elif mode == 'ncc':
        corr_1_to_2 = ncc_correspondence(img_1, img_2, corners_xy_1, corners_xy_2, threshold=0, kernel_size=corr_kernel_size)

    draw_correspondence(img_bgr_1, img_bgr_2, corr_1_to_2, corners_xy_1, corners_xy_2, dest_img_path=dest_img_path)



if __name__ == "__main__":
    img_paths = [["/Users/aartighatkesar/Documents/Harris-Corner-Detector/input_imgs/pair1/1.jpg",
                "/Users/aartighatkesar/Documents/Harris-Corner-Detector/input_imgs/pair1/2.jpg"],
                ["/Users/aartighatkesar/Documents/Harris-Corner-Detector/input_imgs/pair2/truck1.jpg",
                 "/Users/aartighatkesar/Documents/Harris-Corner-Detector/input_imgs/pair2/truck2.jpg"]]

    lst_sigma = [0.707, 1, 1.414, 2]
    nms_kernel_size = 25
    k = 0.06
    thresh = 0.75
    max_no_corners = 500
    corr_kernel_size = 21
    mode = ['ssd', 'ncc']

    max_lines=100  # Max no of lines to plot on image showing correspondence

    results_dir = '/Users/aartighatkesar/Documents/Harris-Corner-Detector/results'

    for img_path in img_paths:
        print(img_path)
        _, fname_1 = os.path.split(img_path[0])
        _, fname_2 = os.path.split(img_path[1])


        for sigma in lst_sigma:
            for m in mode:

                fname = "{}_{}_sig_{}_m_{}.jpg".format(fname_1.split('.')[0], fname_2.split('.')[0], sigma, m)

                print(" #### Processing {} ####".format(fname))

                dest_img_path = os.path.join(results_dir, fname)
                run_main(img_path, dest_img_path=dest_img_path, nms_kernel_size=nms_kernel_size,
                         sigma=sigma, k=k, thresh=thresh, max_no_corners=-1, mode=m, corr_kernel_size=corr_kernel_size)

