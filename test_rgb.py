# test phase
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import os
from torch.autograd import Variable
from net import Net_G
import utils
import numpy as np
import torch.nn.functional as F
import time
import numpy as np    
import cv2    
import time
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from fvcore.nn import FlopCountAnalysis

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model_path = "bert-base-uncased"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    bert_model = AutoModel.from_pretrained(bert_model_path).to(device)
    bert_model.eval()

    test_path = ""

    in_c = 2
    out_c = 1
    model_path = ""
    result = []

    with torch.no_grad():

        model = load_model(model_path, in_c, out_c, device)
        output_path = ''   
        if os.path.exists(output_path) is False:
            os.mkdir(output_path)
        for i in range(250):

            index = i+1
            mri_t1_path = test_path + f'MRI-T1/{index}.png'
            mri_t2_path = test_path + f'MRI-T2/{index}.png'
            pathology_description = read_description(
                [test_path + f"/Pathology_Orders/{index}.txt", test_path + f"/Pathology_Orders/{index}_5.txt", test_path + f"/text/{index}_5.txt"],
                "pathology report describing tissue appearance and lesion semantics",
            )
            ultrasound_description = read_description(
                [test_path + f"/Ultrasound_Orders/{index}.txt", test_path + f"/Ultrasound_Orders/{index}_5.txt", test_path + f"/text/{index}_5.txt"],
                "ultrasound report describing anatomical structure and diagnostic cues",
            )
     
            elapsed_time = run_demo(device, bert_tokenizer, bert_model, model, mri_t1_path, mri_t2_path, pathology_description, ultrasound_description, output_path, index)
            result.append(elapsed_time)
        avg_time = np.mean(result)
    print("Avg Time: {:.4f}s\n".format(avg_time))
    print('Done......')


def load_model(path, input_nc, output_nc, device):

    TextFusionNet_model = Net_G()

    TextFusionNet_model.load_state_dict(torch.load(path, map_location=device))
    TextFusionNet_model.to(device)

    para = sum([np.prod(list(p.size())) for p in TextFusionNet_model.parameters()])
    type_size = 4
    x = torch.randn(1, 1, 640, 480).cuda()
    y = torch.randn(1, 1, 640, 480).cuda()
    z = torch.randn(1, 768).cuda()
    flops = FlopCountAnalysis(TextFusionNet_model, (x, y, z))
    print("FLOPs(G): %.4f" % (flops.total()/1e9))

    print('Model {} : params: {:4f}M'.format(TextFusionNet_model._get_name(), para / 1000/1000))

    TextFusionNet_model.eval()

    return TextFusionNet_model


def read_description(paths, fallback):
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
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



def run_demo(device, bert_tokenizer, bert_model, model, mri_t1_path, mri_t2_path, pathology_description, ultrasound_description, output_path_root, index):

    mri_t2_img = cv2.imread(mri_t2_path, cv2.IMREAD_GRAYSCALE)
    mri_t1_img = Image.open(mri_t1_path).convert("RGB")
    H, W = mri_t2_img.shape
    h, w = mri_t2_img.shape

    new_h = (h // 16) * 16
    new_w = (w // 16) * 16
    mri_t2_img = cv2.resize(mri_t2_img, (new_w, new_h))
    mri_t1_img = mri_t1_img.resize((new_w, new_h))

    mri_t1_y, mri_t1_cb, mri_t1_cr = rgb_to_ycbcr(mri_t1_img)
    
    pathology_text_features = encode_bert_text(bert_model, bert_tokenizer, [pathology_description], device)
    ultrasound_text_features = encode_bert_text(bert_model, bert_tokenizer, [ultrasound_description], device)
    
    mri_t2_img = mri_t2_img / 255.0
    mri_t1_img = mri_t1_y / 255.0
    
    h = mri_t1_img.shape[0]
    w = mri_t1_img.shape[1]
    
    mri_t2_patches = np.resize(mri_t2_img, [1, 1, h, w])
    mri_t1_patches = np.resize(mri_t1_img, [1, 1, h, w])
    
    mri_t2_patches = torch.from_numpy(mri_t2_patches).float()
    mri_t1_patches = torch.from_numpy(mri_t1_patches).float()
    
    
    mri_t2_patches = mri_t2_patches.cuda(device)
    mri_t1_patches = mri_t1_patches.cuda(device)
    model = model.cuda(device)
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
    
    file_name = f'{index}.png'
    if os.path.exists(output_path_root) is False:
        os.mkdir(output_path_root)
    output_path = os.path.join(output_path_root, file_name)

    # fuseImage.save(output_path)

    print(output_path)
    return elapsed_time

if __name__ == '__main__':
    main()
