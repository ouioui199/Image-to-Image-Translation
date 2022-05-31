import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class Metrics():
    def __init__(self):
        pass

    def _assert_image_shapes_equal(self, org_img: np.ndarray, pred_img: np.ndarray, metric: str):

        msg = (
            f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
            f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}"
        )

        assert org_img.shape == pred_img.shape, msg

    def rmse(self, org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 4095) -> float:

        self._assert_image_shapes_equal(org_img, pred_img, "RMSE")

        rmse_bands = []
        for i in range(org_img.shape[2]):
            dif = np.subtract(org_img[:, :, i], pred_img[:, :, i])
            m = np.mean(np.square(dif / max_p))
            s = np.sqrt(m)
            rmse_bands.append(s)

        return np.mean(rmse_bands)

    def psnr(self, org_img: np.ndarray, pred_img: np.ndarray) -> float:

        self._assert_image_shapes_equal(org_img, pred_img, "PSNR")

        return peak_signal_noise_ratio(org_img, pred_img)

    def ssim(self, org_img: np.ndarray, pred_img: np.ndarray) -> float:

        self._assert_image_shapes_equal(org_img, pred_img, "SSIM")

        return structural_similarity(org_img, pred_img, multichannel=True, full=True)

    def calculate_SSIM(self,img1,img2):
        (score, _) = structural_similarity(img1,img2, full=True, multichannel=True)
        return score

    def sliding_window(self, image: np.ndarray, stepSize: int, windowSize: int):

        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):

                yield (x, y, image[y : y + windowSize[1], x : x + windowSize[0]])

    def uiq(self, org_img: np.ndarray, pred_img: np.ndarray, step_size: int = 1, window_size: int = 8):

        self._assert_image_shapes_equal(org_img, pred_img, "UIQ")

        org_img = org_img.astype(np.float32)
        pred_img = pred_img.astype(np.float32)

        q_all = []
        for (x, y, window_org), (x, y, window_pred) in zip(
            self.sliding_window(
                org_img, stepSize=step_size, windowSize=(window_size, window_size)
            ),
            self.sliding_window(
                pred_img, stepSize=step_size, windowSize=(window_size, window_size)
            ),
        ):
            if window_org.shape[0] != window_size or window_org.shape[1] != window_size:
                continue

            for i in range(org_img.shape[2]):
                org_band = window_org[:, :, i]
                pred_band = window_pred[:, :, i]
                org_band_mean = np.mean(org_band)
                pred_band_mean = np.mean(pred_band)
                org_band_variance = np.var(org_band)
                pred_band_variance = np.var(pred_band)
                org_pred_band_variance = np.mean(
                    (org_band - org_band_mean) * (pred_band - pred_band_mean)
                )

                numerator = 4 * org_pred_band_variance * org_band_mean * pred_band_mean
                denominator = (org_band_variance + pred_band_variance) * (
                    org_band_mean ** 2 + pred_band_mean ** 2
                )

                if denominator != 0.0:
                    q = numerator / denominator
                    q_all.append(q)

        if not np.any(q_all):
            raise ValueError(
                f"Window size ({window_size}) is too big for image with shape "
                f"{org_img.shape[0:2]}, please use a smaller window size."
            )

        return np.mean(q_all)
