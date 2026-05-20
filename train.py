import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModel, AutoTokenizer
from net import Net_D, Net_G


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train PTS-GAN for RGB/IR image fusion.")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root containing ir/, vis/ and optional text/.")
    parser.add_argument("--ir-dir", type=str, default="ir", help="Infrared image subdirectory.")
    parser.add_argument("--vis-dir", type=str, default="vis", help="Visible RGB image subdirectory.")
    parser.add_argument("--text-dir", type=str, default="text", help="Shared text prompt fallback subdirectory.")
    parser.add_argument("--vis-text-dir", type=str, default="vis_text", help="Visible image text prompt subdirectory.")
    parser.add_argument("--ir-text-dir", type=str, default="ir_text", help="Infrared image text prompt subdirectory.")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--resume-g", type=str, default="", help="Optional generator checkpoint.")
    parser.add_argument("--resume-d-ir", type=str, default="", help="Optional IR discriminator checkpoint.")
    parser.add_argument("--resume-d-vis", type=str, default="", help="Optional VIS discriminator checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1, help="Keep this as 1 unless net.py is refactored for batched text features.")
    parser.add_argument("--patch-size", type=int, default=256, help="Random crop size, must be divisible by 16.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda-intensity", type=float, default=20.0)
    parser.add_argument("--lambda-gradient", type=float, default=10.0)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--default-text", type=str, default="infrared and visible image fusion scene")
    parser.add_argument("--default-vis-text", type=str, default="visible light scene with texture and color details")
    parser.add_argument("--default-ir-text", type=str, default="infrared thermal scene with salient heat targets")
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased", help="Local BERT directory or Hugging Face model name.")
    parser.add_argument("--text-max-length", type=int, default=77)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    return parser.parse_args()


def image_files(directory):
    return sorted([p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTS])


def find_pairs(data_root, ir_dir, vis_dir):
    ir_root = Path(data_root) / ir_dir
    vis_root = Path(data_root) / vis_dir
    if not ir_root.exists():
        raise FileNotFoundError(f"IR directory not found: {ir_root}")
    if not vis_root.exists():
        raise FileNotFoundError(f"VIS directory not found: {vis_root}")

    vis_by_stem = {p.stem: p for p in image_files(vis_root)}
    pairs = []
    for ir_path in image_files(ir_root):
        vis_path = vis_by_stem.get(ir_path.stem)
        if vis_path is not None:
            pairs.append((ir_path, vis_path, ir_path.stem))

    if not pairs:
        raise RuntimeError(f"No paired images found by filename stem in {ir_root} and {vis_root}.")
    return pairs


def read_prompt(text_roots, stem, fallback):
    for text_root in text_roots:
        candidates = [text_root / f"{stem}.txt", text_root / f"{stem}_5.txt"]
        for path in candidates:
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    return text.splitlines()[0]
    return fallback


def pil_to_gray_tensor(image):
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.float().div(255.0)
    return data.view(image.size[1], image.size[0]).unsqueeze(0)


class FusionDataset(Dataset):
    def __init__(self, args):
        self.pairs = find_pairs(args.data_root, args.ir_dir, args.vis_dir)
        self.text_root = Path(args.data_root) / args.text_dir
        self.vis_text_root = Path(args.data_root) / args.vis_text_dir
        self.ir_text_root = Path(args.data_root) / args.ir_text_dir
        self.patch_size = args.patch_size
        self.default_text = args.default_text
        self.default_vis_text = args.default_vis_text
        self.default_ir_text = args.default_ir_text
        if self.patch_size % 16 != 0:
            raise ValueError("--patch-size must be divisible by 16.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ir_path, vis_path, stem = self.pairs[index]
        ir = Image.open(ir_path).convert("L")
        vis_rgb = Image.open(vis_path).convert("RGB")
        vis_y = vis_rgb.convert("YCbCr").split()[0]

        width = min(ir.width, vis_y.width)
        height = min(ir.height, vis_y.height)
        width = (width // 16) * 16
        height = (height // 16) * 16
        ir = ir.resize((width, height), Image.BICUBIC)
        vis_y = vis_y.resize((width, height), Image.BICUBIC)

        if self.patch_size > 0 and height >= self.patch_size and width >= self.patch_size:
            top = torch.randint(0, height - self.patch_size + 1, (1,)).item()
            left = torch.randint(0, width - self.patch_size + 1, (1,)).item()
            box = (left, top, left + self.patch_size, top + self.patch_size)
            ir = ir.crop(box)
            vis_y = vis_y.crop(box)

        vis_prompt = read_prompt([self.vis_text_root, self.text_root], stem, self.default_vis_text or self.default_text)
        ir_prompt = read_prompt([self.ir_text_root, self.text_root], stem, self.default_ir_text or self.default_text)
        return {
            "ir": pil_to_gray_tensor(ir),
            "vis": pil_to_gray_tensor(vis_y),
            "vis_text": vis_prompt,
            "ir_text": ir_prompt,
            "stem": stem,
        }


def gradient_map(x):
    kernel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    kernel_y = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    grad_x = F.conv2d(x, kernel_x, padding=1)
    grad_y = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


def downsample_like(x, target):
    return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)


def discriminator_loss(discriminator, real_source, fake_outputs, text_features, criterion):
    fake_1, fake_2, fake_3 = [out.detach() for out in fake_outputs]
    real_1 = real_source
    real_2 = downsample_like(real_source, fake_2)
    real_3 = downsample_like(real_source, fake_3)

    pred_real = discriminator(real_1, real_2, real_3, text_features)
    pred_fake = discriminator(fake_1, fake_2, fake_3, text_features)
    loss_real = criterion(pred_real, torch.ones_like(pred_real))
    loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
    return 0.5 * (loss_real + loss_fake)


def generator_adv_loss(discriminator, fake_outputs, text_features, criterion):
    pred_fake = discriminator(fake_outputs[0], fake_outputs[1], fake_outputs[2], text_features)
    return criterion(pred_fake, torch.ones_like(pred_fake))


def encode_bert_text(bert_model, tokenizer, texts, device, max_length):
    encoded = tokenizer(
        list(texts),
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


def fusion_reconstruction_loss(outputs, ir, vis, lambda_intensity, lambda_gradient):
    out_1, out_2, out_3 = outputs
    target = torch.max(ir, vis)
    target_2 = downsample_like(target, out_2)
    target_3 = downsample_like(target, out_3)

    loss_intensity = (
        F.l1_loss(out_1, target)
        + 0.5 * F.l1_loss(out_2, target_2)
        + 0.25 * F.l1_loss(out_3, target_3)
    )

    target_grad = torch.max(gradient_map(ir), gradient_map(vis))
    target_grad_2 = downsample_like(target_grad, out_2)
    target_grad_3 = downsample_like(target_grad, out_3)
    loss_gradient = (
        F.l1_loss(gradient_map(out_1), target_grad)
        + 0.5 * F.l1_loss(gradient_map(out_2), target_grad_2)
        + 0.25 * F.l1_loss(gradient_map(out_3), target_grad_3)
    )
    return lambda_intensity * loss_intensity + lambda_gradient * loss_gradient, loss_intensity, loss_gradient


def load_state(model, path, device, key=None):
    if path:
        state = torch.load(path, map_location=device)
        if isinstance(state, dict):
            if key is not None and key in state:
                state = state[key]
            elif "model" in state:
                state = state["model"]
        model.load_state_dict(state)


def save_checkpoint(save_dir, epoch, generator, d_ir, d_vis, opt_g, opt_d_ir, opt_d_vis):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "d_ir": d_ir.state_dict(),
        "d_vis": d_vis.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d_ir": opt_d_ir.state_dict(),
        "opt_d_vis": opt_d_vis.state_dict(),
    }
    torch.save(payload, Path(save_dir) / f"pts_gan_epoch_{epoch:04d}.pth")
    torch.save(generator.state_dict(), Path(save_dir) / "net_g_latest.pth")


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("The current net.py text-conditioning code assumes batch size 1. Use --batch-size 1.")

    device = torch.device(args.device)
    dataset = FusionDataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    generator = Net_G(text_ch=768).to(device)
    d_ir = Net_D(text_ch=768).to(device)
    d_vis = Net_D(text_ch=768).to(device)
    load_state(generator, args.resume_g, device, key="generator")
    load_state(d_ir, args.resume_d_ir, device, key="d_ir")
    load_state(d_vis, args.resume_d_vis, device, key="d_vis")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    bert_model = AutoModel.from_pretrained(args.bert_model).to(device)
    bert_model.eval()
    for parameter in bert_model.parameters():
        parameter.requires_grad_(False)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d_ir = torch.optim.Adam(d_ir.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    opt_d_vis = torch.optim.Adam(d_vis.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    adv_criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        generator.train()
        d_ir.train()
        d_vis.train()

        for step, batch in enumerate(loader, start=1):
            ir = batch["ir"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)
            with torch.no_grad():
                vis_text_features = encode_bert_text(
                    bert_model, tokenizer, batch["vis_text"], device, args.text_max_length
                )
                ir_text_features = encode_bert_text(
                    bert_model, tokenizer, batch["ir_text"], device, args.text_max_length
                )

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    fake_outputs = generator(
                        vis=vis,
                        ir=ir,
                        vis_text_features=vis_text_features,
                        ir_text_features=ir_text_features,
                    )

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                loss_d_ir = discriminator_loss(d_ir, ir, fake_outputs, ir_text_features, adv_criterion)
                loss_d_vis = discriminator_loss(d_vis, vis, fake_outputs, vis_text_features, adv_criterion)
                loss_d = loss_d_ir + loss_d_vis

            opt_d_ir.zero_grad(set_to_none=True)
            opt_d_vis.zero_grad(set_to_none=True)
            scaler.scale(loss_d).backward()
            scaler.step(opt_d_ir)
            scaler.step(opt_d_vis)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                fake_outputs = generator(
                    vis=vis,
                    ir=ir,
                    vis_text_features=vis_text_features,
                    ir_text_features=ir_text_features,
                )
                loss_rec, loss_intensity, loss_gradient = fusion_reconstruction_loss(
                    fake_outputs,
                    ir,
                    vis,
                    args.lambda_intensity,
                    args.lambda_gradient,
                )
                loss_adv = (
                    generator_adv_loss(d_ir, fake_outputs, ir_text_features, adv_criterion)
                    + generator_adv_loss(d_vis, fake_outputs, vis_text_features, adv_criterion)
                )
                loss_g = loss_rec + args.lambda_adv * loss_adv

            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()

            if step % args.log_every == 0 or step == 1:
                print(
                    f"epoch {epoch:03d}/{args.epochs:03d} "
                    f"step {step:04d}/{len(loader):04d} "
                    f"G {loss_g.item():.4f} D {loss_d.item():.4f} "
                    f"adv {loss_adv.item():.4f} int {loss_intensity.item():.4f} grad {loss_gradient.item():.4f}"
                )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(args.save_dir, epoch, generator, d_ir, d_vis, opt_g, opt_d_ir, opt_d_vis)


if __name__ == "__main__":
    main()
