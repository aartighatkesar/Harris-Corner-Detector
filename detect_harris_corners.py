import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def load_image_gray(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

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

    Ix = convolve2d(img, filter_x, mode='same', boundary='fill', fillvalue=0)

    Iy = convolve2d(img, filter_y, mode='same', boundary='fill', fillvalue=0)

    return Ix, Iy


def create_corner_response_matrix(img, sigma, k):
    """

    :param Ix:
    :param Iy:
    :param sigma:
    :param k: empirical constant k = 0.04 to 0.06
    :return:
    """
    # Create Haar filters:
    filter_x, filter_y = create_Haar_wavelet_xy(sigma)

    # Compute derivatives in x and y using Haar wavelets
    Ix, Iy = compute_derivatives(img, filter_x, filter_y)

    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(Ix, cmap='gray', vmin=np.amin(Ix), vmax=np.amax(Ix))
    plt.title("x-gradient")

    plt.subplot(1, 3, 3)
    plt.imshow(Iy, cmap='gray', vmin=np.amin(Iy), vmax=np.amax(Iy))
    plt.title("y-gradient")

    Ix = Ix - np.mean(Ix)
    Iy = Iy - np.mean(Iy)

    # Given sigma, window w(x,y) is N x N where N is smallest odd integer greater than 5 * sigma
    win_size = int(5 * sigma)

    if win_size % 2:
        win_size += 1
    else:
        win_size += 2


    win = np.ones((win_size, win_size))  # 5sig x 5sig neighbourhood

    Ix_2_win = convolve2d(Ix*Ix, win, mode='same', boundary='fill', fillvalue=0)  # w(x, y) * (Ix_square)

    Iy_2_win = convolve2d(Iy*Iy, win, mode='same', boundary='fill', fillvalue=0)  # w(x, y) * (Iy_square)

    Ix_Iy_win = convolve2d(Ix*Iy, win, mode='same', boundary='fill', fillvalue=0)  # w(x, y) * (Ix times Iy)

    corner_resp = np.zeros_like(Ix_2_win)

    h, w = corner_resp.shape

    for r in range(h):
        for c in range(w):
            mat = np.array([[Ix_2_win[r, c], Ix_Iy_win[r, c]], [Ix_Iy_win[r, c], Iy_2_win[r, c]]])
            det_corner = np.linalg.det(mat)
            trace_corner = np.trace(mat)

            corner_resp[r, c] = det_corner - k * trace_corner**2

    print("max corner resp : {}".format(np.amax(corner_resp)))
    print("min corner resp : {}".format(np.amin(corner_resp)))

    avg = np.mean(corner_resp[corner_resp > 0])
    corner_resp[np.where(corner_resp < 0.5 * avg)] = 0

    corner_pts = np.zeros((h, w))
    avg = np.mean(corner_resp[corner_resp > 0])
    corner_pts[np.where(corner_resp > 0.5 * avg)] = 255


    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("Original image")

    plt.subplot(1, 2, 2)
    plt.imshow(corner_pts, cmap="gray", vmin=0, vmax=255)
    plt.title("Corner response heat map")

    plt.show()

    return corner_resp


def plot_corners(img_clr, corner_resp):

    rows, cols = np.where(corner_resp != 0)


    for i in range(rows.shape[0]):
        cv2.circle(img_clr, (cols[i], rows[i]), radius=3, color=[255, 0, 0], thickness=-1)

    cv2.imshow('Corners', img_clr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def local_non_maximal_suppression(kernel_size, corner_resp):
    pad = int(kernel_size/2)

    corner_resp = np.pad(corner_resp, ((pad, pad), (pad, pad)), mode="constant")

    h, w = corner_resp.shape

    for i in range(pad, h-pad):
        for j in range(pad, w-pad):
            sl = corner_resp[i - pad: i + pad + 1, j - pad: j + pad + 1]
            max_Val = np.amax(sl)
            sl[sl<max_Val] = 0

    corner_resp = corner_resp[pad:h-pad, pad:w-pad]

    return corner_resp






def adaptive_non_maximal_suppression():
    pass


def run_main(img_path, sigma, k):

    img_bgr = load_image_gray(img_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    corner_response = create_corner_response_matrix(img_gray, sigma, k)
    kernel_size = 5

    corner_response = local_non_maximal_suppression(kernel_size, corner_response)

    out = corner_response.copy()
    out[out!=0] = 255
    # plt.figure()
    # plt.imshow(out, cmap="gray", vmin=0, vmax=255)
    # plt.show()
    plot_corners(img_bgr, corner_response)












if __name__ == "__main__":

    sigma = 0.7
    k = 0.05
    # img_path = "/Users/aartighatkesar/Documents/Harris-Corner-Detector/input_imgs/pair1/1.jpg"
    # img_path = "/Users/aartighatkesar/Desktop/5x5checkerboard.png"
    img_path = "/Users/aartighatkesar/Documents/Harris-Corner-Detector/input_imgs/pair2/truck2.jpg"

    run_main(img_path, sigma, k)




