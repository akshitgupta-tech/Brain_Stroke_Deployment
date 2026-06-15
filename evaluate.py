import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def draw_visualization(img_tensor, landmarks, img_path,
                       true_label, pred_label, prob, save_path):

    img = denormalize(img_tensor)
    h, w, _ = img.shape  # 224,224

    # ---- Get original image size ----
    orig = Image.open(img_path)
    orig_w, orig_h = orig.size

    # ---- Correct scaling ----
    scale_x = w / orig_w
    scale_y = h / orig_h

    lm = landmarks.copy()
    lm[:, 0] *= scale_x
    lm[:, 1] *= scale_y

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Bounding box
    x_min, y_min = lm.min(axis=0).astype(int)
    x_max, y_max = lm.max(axis=0).astype(int)
    draw.rectangle([x_min, y_min, x_max, y_max], outline="cyan", width=4)

    # Landmarks
    for (x, y) in lm:
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill='lime')

    # Text
    text = [
        f"Pred: {'STROKE' if pred_label==1 else 'NO STROKE'} ({prob:.2f})",
        f"True: {'STROKE' if true_label==1 else 'NO STROKE'}",
        "CORRECT" if pred_label==true_label else "WRONG"
    ]

    y_offset = 10
    for line in text:
        # outline text for visibility (no background box)
        draw.text((15, y_offset + 5), line, fill="black")
        draw.text((14, y_offset + 4), line,
                fill="green" if pred_label==true_label else "red")
        y_offset += 28


    pil_img.save(save_path)


def evaluate_model(model, test_loader, device, save_dir):
    model.eval()
    save_dir = Path(save_dir)
    vis_dir = save_dir / "droopy_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, graphs, targets, landmarks, paths in test_loader:

            images = images.to(device)
            graphs = graphs.to(device)

            logits, _, _, _ = model(images, graphs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for i in range(len(images)):
                draw_visualization(
                    images[i].cpu(),
                    landmarks[i].numpy(),
                    paths[i],
                    int(targets[i]),
                    int(preds[i]),
                    float(probs[i]),
                    vis_dir / f"sample_{len(all_preds)+i}.png"
                )

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    print(classification_report(all_targets, all_preds,
                                target_names=["No Stroke", "Stroke"]))


def run_evaluation_only(model_path, test_loader, device, save_dir):
    from models.ensemble_model import FacialParalysisEnsemble
    import yaml

    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)

    model = FacialParalysisEnsemble(config)
    # checkpoint = torch.load(model_path, map_location=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Loaded model from epoch {checkpoint.get('epoch')}")

    evaluate_model(model, test_loader, device, save_dir)
