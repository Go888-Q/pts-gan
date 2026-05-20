import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image

from net import Net_G

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()


def parse_args():
    parser = argparse.ArgumentParser(description="Run MRI-T1/MRI-T2 fusion test with a trained PTS-GAN generator.")
    parser.add_argument("--data-root", type=str, default="/data/wangjiaqi/fusion")
    parser.add_argument("--mri-t1-dir", type=str, default="MRI-T1")
    parser.add_argument("--mri-t2-dir", type=str, default="MRI-T2")
    parser.add_argument("--pathology-text-dir", type=str, default="Pathology_Orders")
    parser.add_argument("--ultrasound-text-dir", type=str, default="Ultrasound_Orders")
    parser.add_argument("--shared-text-dir", type=str, default="text")
    parser.add_argument("--model-path", type=str, default="checkpoints/net_g_latest.pth")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased")
    parser.add_argument("--text-max-length", type=int, default=77)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    bert_model = AutoModel.from_pretrained(args.bert_model).to(device)
    bert_model.eval()

    result = []

    with torch.no_grad():
        model = load_model(args.model_path, device)
        pairs = find_pairs(data_root / args.mri_t1_dir, data_root / args.mri_t2_dir)
        for index, (stem, mri_t1_path, mri_t2_path) in enumerate(pairs, start=1):
            pathology_description = read_description(
                [
                    data_root / args.pathology_text_dir / f"{stem}.txt",
                    data_root / args.pathology_text_dir / f"{stem}_5.txt",
                    data_root / args.shared_text_dir / f"{stem}.txt",
                    data_root / args.shared_text_dir / f"{stem}_5.txt",
                ],
                "pathology report describing tissue appearance and lesion semantics",
            )
            ultrasound_description = read_description(
                [
                    data_root / args.ultrasound_text_dir / f"{stem}.txt",
                    data_root / args.ultrasound_text_dir / f"{stem}_5.txt",
                    data_root / args.shared_text_dir / f"{stem}.txt",
                    data_root / args.shared_text_dir / f"{stem}_5.txt",
                ],
                "ultrasound report describing anatomical structure and diagnostic cues",
            )
     
            elapsed_time = run_demo(
                device,
                bert_tokenizer,
                bert_model,
                model,
                mri_t1_path,
                mri_t2_path,
                pathology_description,
                ultrasound_description,
                output_dir,
                stem,
                args.text_max_length,
            )
            result.append(elapsed_time)
        avg_time = np.mean(result)
    print("Avg Time: {:.4f}s\n".format(avg_time))
    print('Done......')


def find_pairs(mri_t1_dir, mri_t2_dir):
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not mri_t1_dir.exists():
        raise FileNotFoundError(f"MRI-T1 directory not found: {mri_t1_dir}")
    if not mri_t2_dir.exists():
        raise FileNotFoundError(f"MRI-T2 directory not found: {mri_t2_dir}")

    mri_t2_by_stem = {
        path.stem: path
        for path in sorted(mri_t2_dir.iterdir())
        if path.is_file() and path.suffix.lower() in image_exts
    }
    pairs = []
    for mri_t1_path in sorted(mri_t1_dir.iterdir()):
        if not mri_t1_path.is_file() or mri_t1_path.suffix.lower() not in image_exts:
            continue
        mri_t2_path = mri_t2_by_stem.get(mri_t1_path.stem)
        if mri_t2_path is not None:
            pairs.append((mri_t1_path.stem, mri_t1_path, mri_t2_path))

    if not pairs:
        raise RuntimeError(f"No paired images found in {mri_t1_dir} and {mri_t2_dir}.")
    return pairs


def load_model(path, device):

    TextFusionNet_model = Net_G()

    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    TextFusionNet_model.load_state_dict(state)
    TextFusionNet_model.to(device)

    para = sum([np.prod(list(p.size())) for p in TextFusionNet_model.parameters()])
    print('Model {} : params: {:4f}M'.format(TextFusionNet_model._get_name(), para / 1000/1000))

    TextFusionNet_model.eval()

    return TextFusionNet_model


def read_description(paths, fallback):
    for path in paths:
        path = Path(path)
        if path.exists():
            with path.open('r', encoding='utf-8', errors='ignore') as f:
                description = f.readline().strip()
                if description:
                    return description
    return fallback


def encode_bert_text(bert_model, tokenizer, texts, device, max_length=77):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = bert_model(**encoded)
    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output.float()
    return outputs.last_hidden_state[:, 0, :].float()
    

def rgb_to_ycbcr(image):
    rgb_array = np.array(image)

    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])

    ycbcr_array = np.dot(rgb_array, transform_matrix.T)

    y_channel = ycbcr_array[:, :, 0]
    cb_channel = ycbcr_array[:, :, 1]
    cr_channel = ycbcr_array[:, :, 2]
    
    y_channel = np.clip(y_channel, 0, 255)
    return y_channel, cb_channel, cr_channel

def ycbcr_to_rgb(y, cb, cr):
    ycbcr_array = np.stack((y, cb, cr), axis=-1)

    transform_matrix = np.array([[1, 0, 1.402],
                                 [1, -0.344136, -0.714136],
                                 [1, 1.772, 0]])
    rgb_array = np.dot(ycbcr_array, transform_matrix.T)
    rgb_array = np.clip(rgb_array, 0, 255)

    rgb_array = np.round(rgb_array).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_array, mode='RGB')

    return rgb_image



def run_demo(device, bert_tokenizer, bert_model, model, mri_t1_path, mri_t2_path, pathology_description, ultrasound_description, output_path_root, stem, text_max_length):

    mri_t2_img = cv2.imread(str(mri_t2_path), cv2.IMREAD_GRAYSCALE)
    if mri_t2_img is None:
        raise FileNotFoundError(f"MRI-T2 image not found or unreadable: {mri_t2_path}")
    mri_t1_img = Image.open(mri_t1_path).convert("RGB")
    H, W = mri_t2_img.shape
    h, w = mri_t2_img.shape

    new_h = (h // 16) * 16
    new_w = (w // 16) * 16
    mri_t2_img = cv2.resize(mri_t2_img, (new_w, new_h))
    mri_t1_img = mri_t1_img.resize((new_w, new_h))

    mri_t1_y, mri_t1_cb, mri_t1_cr = rgb_to_ycbcr(mri_t1_img)
    
    pathology_text_features = encode_bert_text(bert_model, bert_tokenizer, [pathology_description], device, text_max_length)
    ultrasound_text_features = encode_bert_text(bert_model, bert_tokenizer, [ultrasound_description], device, text_max_length)
    
    mri_t2_img = mri_t2_img / 255.0
    mri_t1_img = mri_t1_y / 255.0
    
    h = mri_t1_img.shape[0]
    w = mri_t1_img.shape[1]
    
    mri_t2_patches = np.resize(mri_t2_img, [1, 1, h, w])
    mri_t1_patches = np.resize(mri_t1_img, [1, 1, h, w])
    
    mri_t2_patches = torch.from_numpy(mri_t2_patches).float()
    mri_t1_patches = torch.from_numpy(mri_t1_patches).float()
    
    
    mri_t2_patches = mri_t2_patches.to(device)
    mri_t1_patches = mri_t1_patches.to(device)
    model = model.to(device)
    st = time.time()
    output, _, _ = model(
        mri_t1=mri_t1_patches,
        mri_t2=mri_t2_patches,
        pathology_text_features=pathology_text_features,
        ultrasound_text_features=ultrasound_text_features,
    )
    elapsed_time = time.time() - st
    fuseImage = np.zeros((h, w))
    
    out = output.cpu().numpy()
    
    fuseImage = out[0][0]
    
    fuseImage = fuseImage * 255
    
    fuseImage = ycbcr_to_rgb(fuseImage, mri_t1_cb, mri_t1_cr)
    fuseImage = fuseImage.resize((W,H))
    
    file_name = f'{stem}.png'
    output_path_root.mkdir(parents=True, exist_ok=True)
    output_path = output_path_root / file_name

    fuseImage.save(output_path)

    print(output_path)
    return elapsed_time

if __name__ == '__main__':
    main()
