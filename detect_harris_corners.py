import numpy as np
import cv2
from scipy.signal import convolve2d
import os
import shutil

def load_image(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    h, w, _ = img.shape

    if h > 600 or w > 800:
        print("resizing image to 600 x 800")
        img = cv2.resize(img, (600, 800))

    return img


def create_Haar_wavelet_xy(sigma):
    # Given sigma,
    # create a Haar wavelet of M x M (where M is the smallest even integer greater than 4*sigma)
    # in the direction x and y

    M = int(4 * sigma)

    if M % 2 == 0:
        M += 2
    else:
        M += 1

    filter_x = np.ones((M, M))
    filter_x[:, 0:int(M/2)] = -1

    filter_y = np.ones((M, M))
    filter_y[int(M/2):, :] = -1

    return filter_x, filter_y


def compute_derivatives(img, filter_x, filter_y):
    """

    :param img:
    :param filter_x: x derivative filter
    :param filter_y: y deriv filter
    :return:
    """

    Ix = convolve2d(img, filter_x, mode='same', boundary='fill', fillvalue=0)

    Iy = convolve2d(img, filter_y, mode='same', boundary='fill', fillvalue=0)

    return Ix, Iy


def create_corner_response_matrix(img, sigma, k, thresh):
    """
    :param img: np array for gray image
    :param sigma:
    :param k: empirical constant k = 0.04 to 0.06
    :param thresh: threshold multiplier for average positive corner response used in filtering. thresh would be 0.75 * avg(cornerresp > 0)
    :return:
    """
    # Create Haar filters:
    filter_x, filter_y = create_Haar_wavelet_xy(sigma)

    # Compute derivatives in x and y using Haar wavelets
    Ix, Iy = compute_derivatives(img, filter_x, filter_y)

    # Given sigma, window w(x,y) is N x N where N is smallest odd integer greater than 5 * sigma
    win_size = int(5 * sigma)

    if win_size % 2 == 0:
        win_size += 1
    else:
        win_size += 2


    win = np.ones((win_size, win_size))  # 5sig x 5sig neighbourhood

    Ix_2_win = convolve2d(Ix*Ix, win, mode='same', boundary='fill', fillvalue=0)  # w(x, y) * (Ix_square)

    Iy_2_win = convolve2d(Iy*Iy, win, mode='same', boundary='fill', fillvalue=0)  # w(x, y) * (Iy_square)

    Ix_Iy_win = convolve2d(Ix*Iy, win, mode='same', boundary='fill', fillvalue=0)  # w(x, y) * (Ix times Iy)


    Ix_2_win = Ix_2_win/np.amax(Ix_2_win)
    Iy_2_win = Iy_2_win/np.amax(Iy_2_win)
    Ix_Iy_win = Ix_Iy_win/np.amax(Ix_Iy_win)

    det_corner = Ix_2_win * Iy_2_win - Ix_Iy_win**2
    trace_corner = Ix_2_win + Iy_2_win

    # Corner response
    corner_resp = det_corner - k * (trace_corner**2)

    print("max corner resp : {}".format(np.amax(corner_resp)))
    print("min corner resp : {}".format(np.amin(corner_resp)))

    avg = np.mean(corner_resp[corner_resp > 0])
    corner_resp[np.where(corner_resp < thresh * avg)] = 0

    corner_pts = np.zeros_like(corner_resp)
    avg = np.mean(corner_resp[corner_resp > 0])
    corner_pts[np.where(corner_resp > thresh * avg)] = 255

    return corner_resp


def plot_corners(img_clr, corners, save_img_path):
    """

    :param img_clr: np array color image BGR
    :param corners: numcorners x 2 (col_1 is x, col_2 is y)
    :param save_img_path: destination path to save image
    :return:
    """
    save_img_path = save_img_path.format(corners.shape[0])
    for i in range(corners.shape[0]):
        cv2.circle(img_clr, (corners[i, 0], corners[i, 1]), radius=3, color=[0, 255, 0], thickness=-1)

    cv2.imwrite(save_img_path, img_clr)


def local_non_maximal_suppression(corner_resp, kernel_size=25):
    """
    Pass a window of kernel_size x kernel_size and retain the pixel if it is a local maximum
    :param corner_resp: np array with Harris corner response
    :param kernel_size: Window size for nms. Uses kernerl_size x kernel_size window to compute local maxima
    :return:
    """
    # Making kernel_size odd
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    pad = int(kernel_size/2)

    # Pad for window
    corner_resp = np.pad(corner_resp, ((pad, pad), (pad, pad)), mode="constant")

    nms_corner_resp = np.zeros_like(corner_resp)

    h, w = corner_resp.shape

    for i in range(pad, h-pad):
        for j in range(pad, w-pad):
            max_val = np.amax(corner_resp[i - pad: i + pad + 1, j - pad: j + pad + 1])
            if corner_resp[i, j] == np.amax(corner_resp[i - pad: i + pad + 1, j - pad: j + pad + 1]):
                nms_corner_resp[i, j] = max_val

    # remove padding
    nms_corner_resp = nms_corner_resp[pad:h-pad, pad:w-pad]

    return nms_corner_resp


def get_max_corners(corner_resp, max_no_corners=-1):
    """
    Function to extract coordinates of corners after Harris corner response processed
    :param corner_resp: np array showing Harris corner response after post processing
    :param retain: no of corners or points to retain. If -1, retains all.
    :return:
    """

    r, c = np.where(corner_resp)
    val = corner_resp[r, c]

    corners_xy = np.vstack((c, r)).T  # num_corners x 2 (col_1 is x coord, col_2 is y_coord)

    if max_no_corners > val.size:
        ind = np.argsort(val)[::-1]
    elif max_no_corners > 0:
        ind = np.argsort(val)[::-1][:max_no_corners]
    else:
        ind = np.argsort(val)[::-1]

    print("num corners before: {}".format(corners_xy.shape[0]))
    corners_xy = corners_xy[ind]
    print("num corners after: {}".format(corners_xy.shape[0]))

    return corners_xy


def adaptive_non_maximal_suppression():
    pass


def detect_corners(img_path, dest_img_path, nms_kernel_size, sigma, k=0.06, thresh=0.75, max_no_corners=-1):
    """
    MAIN function to call to get corners arranged as x, y crd
    :param img_path: full path to image whose corners to be detected
    :param dest_img_path: if empty, then does not plot and save result image, else give full path
    :param nms_kernel_size: kernel size to be used during non maximal suppression
    :param sigma: scale
    :param k: empirical constant when computing Harris corner response. Refer to paper. Usually 0.04 ~ 0.06
    :param thresh: threshold ratio i.e corners whose corner response is less than thresh*mean(corner_response) are ignored
    :param max_no_corners: Maximum no of corners to retain. If -1, then all are retained
    :return: corners: nd array N x 2 arranged as x, y coordinate
    """

    # Load image
    img_bgr = load_image(img_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Harris Corner response
    corner_response = create_corner_response_matrix(img_gray, sigma, k, thresh)

    # Non Maximal Suppression
    nms_corner_resp = local_non_maximal_suppression(corner_response, nms_kernel_size)

    # Retain "max_no_corners"
    corners = get_max_corners(nms_corner_resp, max_no_corners)

    if dest_img_path:
        # Plot corners on image
        plot_corners(img_bgr, corners, dest_img_path)

    return corners


if __name__ == "__main__":

    results_dir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    else:
        shutil.rmtree(results_dir)

    # Process all images at once to generate images with corners
    ##### pair ####
    imgs = ["/Users/aartighatkesar/Documents/Harris-Corner-Detector/input_imgs/pair1/1.jpg",
                "/Users/aartighatkesar/Documents/Harris-Corner-Detector/input_imgs/pair1/2.jpg",
                "/Users/aartighatkesar/Documents/Harris-Corner-Detector/input_imgs/pair2/truck1.jpg",
                "/Users/aartighatkesar/Documents/Harris-Corner-Detector/input_imgs/pair2/truck2.jpg"]

    for img_path in imgs:
        print("#########")
        fldr, fname = os.path.split(img_path)
        res_img = os.path.join(results_dir, 'res_' + fname.split('.')[0])
        os.makedirs(res_img)

        lst_sigma = [0.707, 1, 1.414, 2]
        # lst_sigma = [0.707]
        nms_kernel_size = 25
        k = 0.06
        thresh = 0.75
        max_no_corners = 500

        for sigma in lst_sigma:
            dest_img_path = os.path.join(res_img, 'sigma_{}_no_c_{}.jpg'.format(sigma, '{}'))
            detect_corners(img_path, dest_img_path, nms_kernel_size, sigma, k, thresh, max_no_corners)






