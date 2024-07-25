import utils
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt


def get_input(file_name):
    img = Image.open(file_name)
    img = np.asarray(img)
    img = utils.to_mtx(img)
    return img


def to_mtx(img):
    """
    This method just reverse x and y of an image matrix because of the different order of x and y in PIL and Matplotlib library
    """
    H, V, C = img.shape
    mtr = np.zeros((V, H, C), dtype='int')
    for i in range(img.shape[0]):
        mtr[:, i] = img[i]
    return mtr


def get_coef(a, b, n):
    res = []
    b = [b[0], b[1], 1]
    dim = 3
    for i in range(dim):
        curr = [0] * dim * 4
        curr[i] = a[0]
        curr[dim + i] = a[1]
        curr[2 * dim + i] = 1 if i != 2 else 0

        curr[3 * dim + n - 1] = -b[i]
        res.append(curr)

    return res


def getPerspectiveTransform(pts1, pts2):
    A = []
    plen = len(pts1)

    for i in range(plen):
        A += utils.get_coef(pts1[i], pts2[i], i)

    B = [0, 0, -1] * plen
    C = np.linalg.solve(A, B)

    res = np.ones(9)
    res[:8] = C.flatten()[:8]

    return res.reshape(3, -1).T


def showWarpPerspective(dst):
    width, height, _ = dst.shape

    # This part is for denoising the result matrix . You can use this if at first you have filled matrix with zeros
    for i in range(width - 1, -1, -1):
        for j in range(height - 1, -1, -1):
            if dst[i][j][0] == 0 and dst[i][j][1] == 0 and dst[i][j][2] == 0:
                if i + 1 < width and j - 1 >= 0:
                    dst[i][j] = dst[i + 1][j - 1]

    utils.showImage(dst, title='Warp Perspective')


def showImage(image, title, save_file=True):
    final_ans = utils.to_mtx(image)
    final_ans = final_ans.astype(np.uint8)

    plt.title(title)
    plt.imshow(final_ans)

    # save_file = True
    if save_file:
        try:
            os.mkdir('out')
        except OSError:
            pass
        path = os.path.join('out', title + '.jpg')
        plt.savefig(path, bbox_inches='tight')

    plt.show()


def Filter(img, filter_matrix):
    m, n, l = img.shape
    res = np.zeros((m, n, l))
    for i in range(m):
        for j in range(n):
            reshaped = np.reshape(img[i, j, :], newshape=(3,))
            res[i, j, :] = filter_matrix.dot(reshaped)
    return res


def warpPerspective(img, transform_matrix, output_width, output_height):
    """
    TODO : find warp perspective of image_matrix and return it
    :return a (width x height) warped image
    """

    result = np.zeros((output_width, output_height, 3))
    # peymayesh rooye kole pixel ha
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # zarb dakheli ya haman T*[x,y,1]
            matrix = np.dot(transform_matrix, [i, j, 1])
            #x'
            xp = int(matrix[0] / matrix[2])
            #y'
            yp = int(matrix[1] / matrix[2])
            #shart dar mahdoode khorooji boodan
            if xp < output_width and yp < output_height:
                result[xp,yp,:]=img[i,j,:]
    return result[:output_width, :output_height, :]


def grayScaledFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """

    M = np.array([[0.299, 0.587, 0.114], [0.299, 0.587, 0.114], [0.299, 0.587, 0.114]])
    return utils.Filter(img, M)


def crazyFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """

    # tabe asli
    M1 = np.array([[0, 0, 1], [0, 0.5, 0], [0.5, 0.5, 0]])
    img_first = utils.Filter(img, M1)
    # varoone tabe
    # ya M2 = np.linalg.inv(M1)
    M2 = np.array([[0, -2, 2], [0, 2, 0], [1, 0, 0]])
    return utils.Filter(img, M1), utils.Filter(img_first, M2)


def permuteFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """

    M = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    return utils.Filter(img, M)


def scaleImg(img, scale_width, scale_height):
    """
    TODO : Complete this part based on the description in the manual!
    """

    # new height
    h2 = img.shape[1] * scale_height
    # new width
    w2 = img.shape[0] * scale_width
    result = np.zeros((w2, h2, 3))
    # peymayesh rooye kole pixel ha
    for i in range(w2):
      for j in range(h2):
        # x = x * (old scale / (scale * old scale ) ==>  new x = x / scale
        result[i, j, :] = img[int(i / scale_width), int(j / scale_height), :]
    return result

if __name__ == "__main__":
    image_matrix = get_input('pic.jpg')

    # You can change width and height if you want
    width, height = 300, 400

    showImage(image_matrix, title="Input Image")

    # TODO : Find coordinates of four corners of your inner Image ( X,Y format)
    #  Order of coordinates: Upper Left, Upper Right, Down Left, Down Right
    pts1 = np.float32([[240, 10], [590, 175], [250, 975], [625, 900]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    m = getPerspectiveTransform(pts1, pts2)

    warpedImage = warpPerspective(image_matrix, m, width, height)
    showWarpPerspective(warpedImage)

    grayScalePic = grayScaledFilter(warpedImage)
    showImage(grayScalePic, title="Gray Scaled")

    crazyImage, invertedCrazyImage = crazyFilter(warpedImage)
    showImage(crazyImage, title="Crazy Filter")
    showImage(invertedCrazyImage, title="Inverted Crazy Filter")

    scaledImage = scaleImg(warpedImage, 3, 4)
    showImage(scaledImage, title="Scaled Image")

    permuteImage = permuteFilter(warpedImage)
    showImage(permuteImage, title="Permuted Image")