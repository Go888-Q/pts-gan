import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModel, AutoTokenizer
from net import Net_D, Net_G


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train PTS-GAN for MRI-T1/MRI-T2 medical image fusion.")
    parser.add_argument("--data-root", type=str, default="/data/wangjiaqi/fusion", help="Dataset root containing paired image folders and optional text folders.")
    parser.add_argument("--mri-t1-dir", type=str, default="MRI-T1", help="MRI-T1 image subdirectory.")
    parser.add_argument("--mri-t2-dir", type=str, default="MRI-T2", help="MRI-T2 image subdirectory.")
    parser.add_argument("--text-dir", type=str, default="text", help="Shared text prompt fallback subdirectory.")
    parser.add_argument("--pathology-text-dir", type=str, default="Pathology_Orders", help="Pathology report text subdirectory.")
    parser.add_argument("--ultrasound-text-dir", type=str, default="Ultrasound_Orders", help="Ultrasound report text subdirectory.")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--resume-g", type=str, default="", help="Optional generator checkpoint.")
    parser.add_argument("--resume-d-mri-t1", type=str, default="", help="Optional MRI-T1 discriminator checkpoint.")
    parser.add_argument("--resume-d-mri-t2", type=str, default="", help="Optional MRI-T2 discriminator checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size. In DDP this is per GPU; in DataParallel this is global batch size.")
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
    parser.add_argument("--default-text", type=str, default="MRI fusion medical imaging study")
    parser.add_argument("--default-pathology-text", type=str, default="pathology report describing tissue appearance and lesion semantics")
    parser.add_argument("--default-ultrasound-text", type=str, default="ultrasound report describing anatomical structure and diagnostic cues")
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased", help="Local BERT directory or Hugging Face model name.")
    parser.add_argument("--text-max-length", type=int, default=77)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--parallel", type=str, default="auto", choices=["auto", "ddp", "dp", "none"], help="Parallel mode: auto uses DDP under torchrun, otherwise DataParallel when multiple GPUs are requested.")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids to use, for example 0,1,2,3,4,5,6,7.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    return parser.parse_args()


def image_files(directory):
    return sorted([p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTS])


def parse_gpu_ids(gpu_ids):
    return [int(item) for item in gpu_ids.split(",") if item.strip()]


def unwrap_model(model):
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def ddp_is_active():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not ddp_is_active() or dist.get_rank() == 0


def setup_parallel(args):
    requested_gpu_ids = parse_gpu_ids(args.gpu_ids) if torch.cuda.is_available() and args.device.startswith("cuda") else []
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    use_ddp = args.parallel == "ddp" or (args.parallel == "auto" and world_size > 1)
    use_dp = args.parallel == "dp" or (args.parallel == "auto" and world_size == 1 and len(requested_gpu_ids) > 1)

    if args.parallel == "none":
        use_ddp = False
        use_dp = False

    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training needs CUDA GPUs.")
        if world_size <= 1:
            raise RuntimeError("DDP needs torchrun. Example: torchrun --nproc_per_node=8 train.py")
        if len(requested_gpu_ids) < world_size:
            raise ValueError(f"DDP world size is {world_size}, but --gpu-ids only has {requested_gpu_ids}.")
        physical_gpu_id = requested_gpu_ids[local_rank]
        torch.cuda.set_device(physical_gpu_id)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{physical_gpu_id}")
        return {
            "mode": "ddp",
            "device": device,
            "gpu_ids": requested_gpu_ids[:world_size],
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "physical_gpu_id": physical_gpu_id,
            "is_main": rank == 0,
        }

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        available_gpus = torch.cuda.device_count()
        missing_gpus = [gpu_id for gpu_id in requested_gpu_ids if gpu_id >= available_gpus]
        if missing_gpus:
            raise ValueError(f"Requested GPU ids {missing_gpus}, but only {available_gpus} CUDA devices are visible.")
        if not requested_gpu_ids:
            requested_gpu_ids = [0]
        torch.cuda.set_device(requested_gpu_ids[0])
        device = torch.device(f"cuda:{requested_gpu_ids[0]}")
    else:
        requested_gpu_ids = []
        device = torch.device("cpu")
        use_dp = False

    return {
        "mode": "dp" if use_dp else "single",
        "device": device,
        "gpu_ids": requested_gpu_ids,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "physical_gpu_id": requested_gpu_ids[0] if requested_gpu_ids else "cpu",
        "is_main": True,
    }


def find_pairs(data_root, mri_t1_dir, mri_t2_dir):
    mri_t1_root = Path(data_root) / mri_t1_dir
    mri_t2_root = Path(data_root) / mri_t2_dir
    if not mri_t1_root.exists():
        raise FileNotFoundError(f"MRI-T1 directory not found: {mri_t1_root}")
    if not mri_t2_root.exists():
        raise FileNotFoundError(f"MRI-T2 directory not found: {mri_t2_root}")

    mri_t2_by_stem = {p.stem: p for p in image_files(mri_t2_root)}
    pairs = []
    for mri_t1_path in image_files(mri_t1_root):
        mri_t2_path = mri_t2_by_stem.get(mri_t1_path.stem)
        if mri_t2_path is not None:
            pairs.append((mri_t1_path, mri_t2_path, mri_t1_path.stem))

    if not pairs:
        raise RuntimeError(f"No paired images found by filename stem in {mri_t1_root} and {mri_t2_root}.")
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
        self.pairs = find_pairs(args.data_root, args.mri_t1_dir, args.mri_t2_dir)
        self.text_root = Path(args.data_root) / args.text_dir
        self.pathology_text_root = Path(args.data_root) / args.pathology_text_dir
        self.ultrasound_text_root = Path(args.data_root) / args.ultrasound_text_dir
        self.patch_size = args.patch_size
        self.default_text = args.default_text
        self.default_pathology_text = args.default_pathology_text
        self.default_ultrasound_text = args.default_ultrasound_text
        if self.patch_size % 16 != 0:
            raise ValueError("--patch-size must be divisible by 16.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        mri_t1_path, mri_t2_path, stem = self.pairs[index]
        mri_t1_rgb = Image.open(mri_t1_path).convert("RGB")
        mri_t1 = mri_t1_rgb.convert("YCbCr").split()[0]
        mri_t2 = Image.open(mri_t2_path).convert("L")

        width = min(mri_t1.width, mri_t2.width)
        height = min(mri_t1.height, mri_t2.height)
        width = (width // 16) * 16
        height = (height // 16) * 16
        mri_t1 = mri_t1.resize((width, height), Image.BICUBIC)
        mri_t2 = mri_t2.resize((width, height), Image.BICUBIC)

        if self.patch_size > 0 and height >= self.patch_size and width >= self.patch_size:
            top = torch.randint(0, height - self.patch_size + 1, (1,)).item()
            left = torch.randint(0, width - self.patch_size + 1, (1,)).item()
            box = (left, top, left + self.patch_size, top + self.patch_size)
            mri_t1 = mri_t1.crop(box)
            mri_t2 = mri_t2.crop(box)

        pathology_prompt = read_prompt([self.pathology_text_root, self.text_root], stem, self.default_pathology_text or self.default_text)
        ultrasound_prompt = read_prompt([self.ultrasound_text_root, self.text_root], stem, self.default_ultrasound_text or self.default_text)
        return {
            "mri_t1": pil_to_gray_tensor(mri_t1),
            "mri_t2": pil_to_gray_tensor(mri_t2),
            "pathology_text": pathology_prompt,
            "ultrasound_text": ultrasound_prompt,
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


def fusion_reconstruction_loss(outputs, mri_t1, mri_t2, lambda_intensity, lambda_gradient):
    out_1, out_2, out_3 = outputs
    target = torch.max(mri_t1, mri_t2)
    target_2 = downsample_like(target, out_2)
    target_3 = downsample_like(target, out_3)

    loss_intensity = (
        F.l1_loss(out_1, target)
        + 0.5 * F.l1_loss(out_2, target_2)
        + 0.25 * F.l1_loss(out_3, target_3)
    )

    target_grad = torch.max(gradient_map(mri_t1), gradient_map(mri_t2))
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


def save_checkpoint(save_dir, epoch, generator, d_mri_t1, d_mri_t2, opt_g, opt_d_mri_t1, opt_d_mri_t2):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "generator": unwrap_model(generator).state_dict(),
        "d_mri_t1": unwrap_model(d_mri_t1).state_dict(),
        "d_mri_t2": unwrap_model(d_mri_t2).state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d_mri_t1": opt_d_mri_t1.state_dict(),
        "opt_d_mri_t2": opt_d_mri_t2.state_dict(),
    }
    torch.save(payload, Path(save_dir) / f"pts_gan_epoch_{epoch:04d}.pth")
    torch.save(unwrap_model(generator).state_dict(), Path(save_dir) / "net_g_latest.pth")


def main():
    args = parse_args()
    parallel = setup_parallel(args)
    device = parallel["device"]

    dataset = FusionDataset(args)
    sampler = DistributedSampler(dataset, shuffle=True) if parallel["mode"] == "ddp" else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    generator = Net_G(text_ch=768).to(device)
    d_mri_t1 = Net_D(text_ch=768).to(device)
    d_mri_t2 = Net_D(text_ch=768).to(device)
    load_state(generator, args.resume_g, device, key="generator")
    load_state(d_mri_t1, args.resume_d_mri_t1, device, key="d_mri_t1")
    load_state(d_mri_t2, args.resume_d_mri_t2, device, key="d_mri_t2")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    bert_model = AutoModel.from_pretrained(args.bert_model).to(device)
    bert_model.eval()
    for parameter in bert_model.parameters():
        parameter.requires_grad_(False)

    if parallel["mode"] == "ddp":
        generator = DDP(generator, device_ids=[device.index], output_device=device.index)
        d_mri_t1 = DDP(d_mri_t1, device_ids=[device.index], output_device=device.index)
        d_mri_t2 = DDP(d_mri_t2, device_ids=[device.index], output_device=device.index)
    elif parallel["mode"] == "dp":
        generator = nn.DataParallel(generator, device_ids=parallel["gpu_ids"], output_device=parallel["gpu_ids"][0])
        d_mri_t1 = nn.DataParallel(d_mri_t1, device_ids=parallel["gpu_ids"], output_device=parallel["gpu_ids"][0])
        d_mri_t2 = nn.DataParallel(d_mri_t2, device_ids=parallel["gpu_ids"], output_device=parallel["gpu_ids"][0])

    if parallel["mode"] == "ddp":
        active_gpu_count = parallel["world_size"]
        batch_description = f"{args.batch_size} per GPU"
        global_batch_size = args.batch_size * parallel["world_size"]
    elif parallel["mode"] == "dp":
        active_gpu_count = len(parallel["gpu_ids"])
        batch_description = f"{args.batch_size} global, split across {active_gpu_count} GPUs"
        global_batch_size = args.batch_size
    elif device.type == "cuda":
        active_gpu_count = 1
        batch_description = f"{args.batch_size} on GPU {parallel['physical_gpu_id']}"
        global_batch_size = args.batch_size
    else:
        active_gpu_count = 0
        batch_description = f"{args.batch_size} on CPU"
        global_batch_size = args.batch_size

    if parallel["is_main"]:
        print("Training parallel configuration:")
        print(f"  mode: {parallel['mode']}")
        print(f"  requested gpu ids: {parallel['gpu_ids'] if parallel['gpu_ids'] else 'cpu'}")
        print(f"  active GPU count: {active_gpu_count}")
        print(f"  DDP world size: {parallel['world_size']}")
        print(f"  rank/local_rank/current GPU: {parallel['rank']}/{parallel['local_rank']}/{parallel['physical_gpu_id']}")
        print(f"  batch size: {batch_description}")
        print(f"  global batch size: {global_batch_size}")
        print(f"  dataset pairs: {len(dataset)}")
        print(f"  steps per epoch on this process: {len(loader)}")

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d_mri_t1 = torch.optim.Adam(d_mri_t1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    opt_d_mri_t2 = torch.optim.Adam(d_mri_t2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    adv_criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        generator.train()
        d_mri_t1.train()
        d_mri_t2.train()

        for step, batch in enumerate(loader, start=1):
            mri_t1 = batch["mri_t1"].to(device, non_blocking=True)
            mri_t2 = batch["mri_t2"].to(device, non_blocking=True)
            with torch.no_grad():
                pathology_text_features = encode_bert_text(
                    bert_model, tokenizer, batch["pathology_text"], device, args.text_max_length
                )
                ultrasound_text_features = encode_bert_text(
                    bert_model, tokenizer, batch["ultrasound_text"], device, args.text_max_length
                )

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    fake_outputs = generator(
                        mri_t1=mri_t1,
                        mri_t2=mri_t2,
                        pathology_text_features=pathology_text_features,
                        ultrasound_text_features=ultrasound_text_features,
                    )

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                loss_d_mri_t1 = discriminator_loss(d_mri_t1, mri_t1, fake_outputs, pathology_text_features, adv_criterion)
                loss_d_mri_t2 = discriminator_loss(d_mri_t2, mri_t2, fake_outputs, ultrasound_text_features, adv_criterion)
                loss_d = loss_d_mri_t1 + loss_d_mri_t2

            opt_d_mri_t1.zero_grad(set_to_none=True)
            opt_d_mri_t2.zero_grad(set_to_none=True)
            scaler.scale(loss_d).backward()
            scaler.step(opt_d_mri_t1)
            scaler.step(opt_d_mri_t2)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                fake_outputs = generator(
                    mri_t1=mri_t1,
                    mri_t2=mri_t2,
                    pathology_text_features=pathology_text_features,
                    ultrasound_text_features=ultrasound_text_features,
                )
                loss_rec, loss_intensity, loss_gradient = fusion_reconstruction_loss(
                    fake_outputs,
                    mri_t1,
                    mri_t2,
                    args.lambda_intensity,
                    args.lambda_gradient,
                )
                loss_adv = (
                    generator_adv_loss(d_mri_t1, fake_outputs, pathology_text_features, adv_criterion)
                    + generator_adv_loss(d_mri_t2, fake_outputs, ultrasound_text_features, adv_criterion)
                )
                loss_g = loss_rec + args.lambda_adv * loss_adv

            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()

            if parallel["is_main"] and (step % args.log_every == 0 or step == 1):
                print(
                    f"epoch {epoch:03d}/{args.epochs:03d} "
                    f"step {step:04d}/{len(loader):04d} "
                    f"G {loss_g.item():.4f} D {loss_d.item():.4f} "
                    f"adv {loss_adv.item():.4f} int {loss_intensity.item():.4f} grad {loss_gradient.item():.4f}"
                )

        if parallel["is_main"] and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_checkpoint(args.save_dir, epoch, generator, d_mri_t1, d_mri_t2, opt_g, opt_d_mri_t1, opt_d_mri_t2)

    if ddp_is_active():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
