
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import sklearn.metrics as skm
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim
import torch
import collections
from PIL import Image

TARGET_H = 512
TARGET_W = 512

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(str(path))
    if img_BGR is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    img_BGR = img_BGR.astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    return img

class Evaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None): 
        if imgA is None:
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return skm.mutual_info_score(image_F.flatten(), image_A.flatten()) + skm.mutual_info_score(image_F.flatten(),
                                                                                                   image_B.flatten())

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  # MSE
        cls.input_check(image_F, image_A, image_B)
        return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * np.log10(np.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    @classmethod
    def SCD(cls, image_F, image_A, image_B): # The sum of the correlations of differences
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
        return corr1 + corr2

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F)+cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls,ref, dist): # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        # 计算QABF
        deno = np.sum(gA + gB)
        if deno <= 1e-12:
            return 0.0
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return float(np.clip(nume / deno, 0.0, 1.0))

    @classmethod
    def Qabf_getArray(cls,img):
        # Sobel Operator Sobel
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0]=np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls,aA, gA, aF, gF):
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        eps = 1e-12
        max_g = np.maximum(gA, gF)
        min_g = np.minimum(gA, gF)
        GAF = np.divide(min_g, max_g + eps, out=np.zeros_like(gA), where=max_g > eps)

        angle_diff = np.abs(aA - aF)
        angle_diff = np.minimum(angle_diff, math.pi - angle_diff)
        AAF = 1 - angle_diff / (math.pi / 2)
        AAF = np.clip(AAF, 0.0, 1.0)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF* QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return ssim(image_F, image_A, data_range=255) + ssim(image_F, image_B, data_range=255)


def VIFF(image_F, image_A, image_B):
    refA=image_A
    refB=image_B
    dist=image_F

    sigma_nsq = 2
    eps = 1e-10
    numA = 0.0
    denA = 0.0
    numB = 0.0
    denB = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        # Create a Gaussian kernel as MATLAB's
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode='valid')
            refB = convolve2d(refB, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            refA = refA[::2, ::2]
            refB = refB[::2, ::2]
            dist = dist[::2, ::2]

        mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
        mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq_A = mu1A * mu1A
        mu1_sq_B = mu1B * mu1B
        mu2_sq = mu2 * mu2
        mu1A_mu2 = mu1A * mu2
        mu1B_mu2 = mu1B * mu2
        sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1_sq_A
        sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1_sq_B
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A_mu2
        sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B_mu2

        sigma1A_sq[sigma1A_sq < 0] = 0
        sigma1B_sq[sigma1B_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        gA = sigma12_A / (sigma1A_sq + eps)
        gB = sigma12_B / (sigma1B_sq + eps)
        sv_sq_A = sigma2_sq - gA * sigma12_A
        sv_sq_B = sigma2_sq - gB * sigma12_B

        gA[sigma1A_sq < eps] = 0
        gB[sigma1B_sq < eps] = 0
        sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
        sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
        sigma1A_sq[sigma1A_sq < eps] = 0
        sigma1B_sq[sigma1B_sq < eps] = 0

        gA[sigma2_sq < eps] = 0
        gB[sigma2_sq < eps] = 0
        sv_sq_A[sigma2_sq < eps] = 0
        sv_sq_B[sigma2_sq < eps] = 0

        sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
        sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
        gA[gA < 0] = 0
        gB[gB < 0] = 0
        sv_sq_A[sv_sq_A <= eps] = eps
        sv_sq_B[sv_sq_B <= eps] = eps

        numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
        numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
        denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
        denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

    vifpA = numA / denA
    vifpB =numB / denB

    if np.isnan(vifpA):
        vifpA=1
    if np.isnan(vifpB):
        vifpB = 1
    return vifpA+vifpB

def avgGradient(path3):
    image = Image.open(path3).convert('L').resize((TARGET_W, TARGET_H), Image.BILINEAR)
    image = np.array(image)
    width = image.shape[1]
    width = width - 1
    heigt = image.shape[0]
    heigt = heigt - 1
    tmp = 0.0
    for i in range(width):
        for j in range(heigt):
            dx = float(image[i, j + 1]) - float(image[i, j])
            dy = float(image[i + 1, j]) - float(image[i, j])
            ds = math.sqrt((dx * dx + dy * dy) / 2)
            tmp += ds
    imageAG = tmp / (width * heigt)
    return round(imageAG,3)

def IE(path3):
    img = cv2.imread(path3, flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img)
    compare_list = []
    for m in range(1, img.size()[0] - 1):
        for n in range(1, img.size()[0] - 1):
            sum_element = img[m - 1, n - 1] + img[m - 1, n] + img[m - 1, n + 1] + img[m, n - 1] + img[m, n + 1] + img[
                m + 1, n - 1] + img[m + 1, n] + img[m + 1, n + 1]
            sum_element = int(sum_element)
            mean_element = sum_element // 8
            pix = int(img[m, n])
            temp = (pix, mean_element)
            compare_list.append(temp)

    compare_dict = collections.Counter(compare_list)
    H = 0.0
    for freq in compare_dict.values():
        f_n2 = freq / img.size()[0] ** 2
        log_f_n2 = math.log(f_n2)
        h = -(f_n2 * log_f_n2)
        H += h
    return H


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MRI-T1/MRI-T2 fusion results.")
    parser.add_argument("--mri-t1-dir", type=str, default="/data/wangjiaqi/fusion/MRI-T1")
    parser.add_argument("--mri-t2-dir", type=str, default="/data/wangjiaqi/fusion/MRI-T2")
    parser.add_argument("--fused-dir", type=str, default="results")
    parser.add_argument("--output-csv", type=str, default="evaluation_results.csv")
    parser.add_argument("--target-height", type=int, default=512)
    parser.add_argument("--target-width", type=int, default=512)
    parser.add_argument("--extensions", type=str, default=".png,.jpg,.jpeg,.bmp,.tif,.tiff")
    return parser.parse_args()


def natural_key(path):
    stem = Path(path).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def collect_images(directory, extensions):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    suffixes = {ext.lower().strip() for ext in extensions.split(",") if ext.strip()}
    images = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            images[path.stem] = path
    return images


def evaluate_one(mri_t1_path, mri_t2_path, fused_path):
    img_mri_t1 = image_read_cv2(mri_t1_path, "GRAY")
    img_mri_t2 = image_read_cv2(mri_t2_path, "GRAY")
    img_fuse = image_read_cv2(fused_path, "GRAY")

    return {
        "IE": Evaluator.EN(img_fuse),
        "AG": Evaluator.AG(img_fuse),
        "SD": Evaluator.SD(img_fuse),
        "SF": Evaluator.SF(img_fuse),
        "MI": Evaluator.MI(img_fuse, img_mri_t1, img_mri_t2),
        "SCD": Evaluator.SCD(img_fuse, img_mri_t1, img_mri_t2),
        "VIFF": Evaluator.VIFF(img_fuse, img_mri_t1, img_mri_t2),
        "Qabf": Evaluator.Qabf(img_fuse, img_mri_t1, img_mri_t2),
        "SSIM": Evaluator.SSIM(img_fuse, img_mri_t1, img_mri_t2),
    }


def main():
    global TARGET_H, TARGET_W
    args = parse_args()
    TARGET_H = args.target_height
    TARGET_W = args.target_width

    mri_t1_images = collect_images(args.mri_t1_dir, args.extensions)
    mri_t2_images = collect_images(args.mri_t2_dir, args.extensions)
    fused_images = collect_images(args.fused_dir, args.extensions)

    common_ids = sorted(
        set(mri_t1_images) & set(mri_t2_images) & set(fused_images),
        key=natural_key,
    )
    if not common_ids:
        raise RuntimeError(
            "No matched images found. Please make sure MRI-T1, MRI-T2, and fused result images use the same file names."
        )

    metric_names = ["IE", "AG", "SD", "SF", "MI", "SCD", "VIFF", "Qabf", "SSIM"]
    rows = []
    for image_id in common_ids:
        metrics = evaluate_one(mri_t1_images[image_id], mri_t2_images[image_id], fused_images[image_id])
        row = {"id": image_id}
        row.update({name: round(float(metrics[name]), 6) for name in metric_names})
        rows.append(row)
        print(f"{image_id}: " + " ".join([f"{name}={row[name]:.4f}" for name in metric_names]))

    output_csv = Path(args.output_csv)
    if output_csv.parent != Path("."):
        output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id"] + metric_names)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluated {len(rows)} image pairs.")
    print(f"CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()

r"""
# Legacy hard-coded evaluation script kept below for reference only.
csv_rows = []
for i in range(1,20):

    print(i)
    # path1 = r"E:\liulong\repository\WELL_Down\FLFuse-Net4\all\lung\CT\{}.jpg".format(i) ## ## 1036， 1037， clean9248 ,*75
    # path2 = r"E:\liulong\repository\WELL_Down\FLFuse-Net4\all\lung\PET\{}.jpg".format(i)
    # path3 = r"E:\liulong\repository\WELL_Down\FLFuse-Net4\all\lung\OUT\{}.png".format(i)  ## 001, 002， 052, *046

    path1 = r"/home/wangjiaqi/CrossFuseText/test_imgs/MRI-T1/{}.png".format(i)  ## ## 1036， 1037， clean9248 ,*75
    path2 = r"/home/wangjiaqi/CrossFuseText/test_imgs/MRI-T2/{}.png".format(i)
    path3 = r"/home/wangjiaqi/CrossFuseText/full/{}.png".format(i)
    img_ct = image_read_cv2(path1,'GRAY')
    img_pet = image_read_cv2(path2,'GRAY')
    img_fuse = image_read_cv2(path3,'GRAY')
    
    metric_result = np.zeros((8))
    metric_result += np.array([Evaluator.EN(img_fuse), Evaluator.SD(img_fuse)
                                  , Evaluator.SF(img_fuse), Evaluator.MI(img_fuse, img_ct, img_pet)
                                  , Evaluator.SCD(img_fuse, img_ct, img_pet), Evaluator.VIFF(img_fuse, img_ct, img_pet)
                                  , Evaluator.Qabf(img_fuse, img_ct, img_pet), Evaluator.SSIM(img_fuse, img_ct, img_pet)])
    csv_rows.append([
        i,
        IE(path3),
        avgGradient(path3),
        np.round(metric_result[1], 6),
        np.round(metric_result[2], 6),
        np.round(metric_result[3], 6),
        np.round(metric_result[4], 6),
        np.round(metric_result[5], 6),
        np.round(metric_result[6], 6),
        np.round(metric_result[7], 6),
    ])
    print( '信息熵IE:'+'\t'+ str("%.2f"%IE(path3)) + '\n'
          + '平均梯度AG:'+'\t'+ str(avgGradient(path3)) + '\n'
          + "标准差SD:"+ '\t'+  str(np.round(metric_result[1], 2)) + '\n'
          + "空间频率SF:"+ '\t'+  str(np.round(metric_result[2], 2)) + '\n'
          + "互信息MI:"+ '\t'+  str(np.round(metric_result[3], 2)) + '\n'
          + "差异相关和SCD:"+ '\t'+ str(np.round(metric_result[4], 2)) + '\n'
          + "视觉信息保真度VIFF:"+ '\t'+ str(np.round(metric_result[5], 2)) + '\n'
          + "Qabf:"+ '\t'+ str(np.round(metric_result[6], 2)) + '\n'
          + "结构相似指数测度SSIM:"+ '\t'+ str(np.round(metric_result[7], 2))
           )

csv_save_path = "evaluation_results.csv"
with open(csv_save_path, "w", encoding="utf-8") as f:
    f.write("id,IE,AG,SD,SF,MI,SCD,VIFF,Qabf,SSIM\n")
    for row in csv_rows:
        f.write(",".join([str(x) for x in row]) + "\n")

print("CSV saved to:", csv_save_path)
"""
