# src/train.py
from __future__ import annotations
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from config import load_config, ensure_dirs
from utils.common import set_seed, device_auto, save_json
from utils.logging import get_logger
from utils.metrics import masked_mse, r2_last_global
from models.deeplobv import DeepLOBv
from data.uam_threeway_prep import ThreeWayConfig, make_threeway_loaders

from torch import amp
from torch.amp import GradScaler


# ---- 路径辅助函数 ----
def project_root_from(file: str) -> Path:
    """给定当前脚本 __file__，返回项目根目录 (src/ 的上一级)."""
    return Path(file).resolve().parents[1]

def data_dir(file: str) -> Path:
    """返回项目下 data/interim 目录."""
    return project_root_from(file) / "data" / "interim"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="覆盖默认输出目录（可选）")
    args = parser.parse_args()

    # ---- 加载配置 ----
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg["seed"])
    device = device_auto()

    # ---- 路径 (相对化) ----
    root = project_root_from(__file__)
    train_path = data_dir(__file__) / "training.parquet"
    test_path  = data_dir(__file__) / "testing.parquet"

    training_df = pd.read_parquet(train_path)
    testing_df  = pd.read_parquet(test_path)

    # ---- DataLoader ----
    dc = cfg["data"]
    twc = ThreeWayConfig(
        date_col=dc["date_col"], time_col=dc["time_col"],
        symbol_col=dc["symbol_col"], y_col=dc["y_col"],
        inner_split_date=dc["inner_split_date"],
        L_days=dc["L_days"], bars_per_day=dc["bars_per_day"],
        min_valid_ratio=dc["min_valid_ratio"],
        use_sparse=dc["use_sparse"], add_delta_g_feature=dc["add_delta_g_feature"],
        delta_cap=dc["delta_cap"], K_per_symbol=dc["K_per_symbol"],
        batch_size=dc["batch_size"], num_workers=dc["num_workers"],
        pin_memory=dc["pin_memory"],
    )
    train_loader, inner_test_loader, outer_test_loader, meta = make_threeway_loaders(
        training_df, testing_df, twc
    )

    in_features = meta["X"].shape[1]
    T, L = meta["T"], meta["L"]

    logger = get_logger("train", root / cfg["paths"]["logs_dir"])
    logger.info(f"in_features={in_features}, T={T}, L={L}, "
                f"train_windows={len(train_loader.dataset)}, "
                f"inner_test_windows={len(inner_test_loader.dataset)}, "
                f"outer_test_windows={len(outer_test_loader.dataset)}")

    # ---- 模型 / 优化器 / 调度 ----
    model = DeepLOBv(in_features=in_features, **cfg["model"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    sch_cfg = cfg["train"]["scheduler"]
    scheduler = None
    if sch_cfg["name"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=sch_cfg["t_max"], eta_min=sch_cfg["min_lr"]
        )
    elif sch_cfg["name"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=sch_cfg["factor"],
            patience=sch_cfg["patience"], min_lr=sch_cfg["min_lr"]
        )

    scaler = GradScaler(enabled=cfg["train"]["amp"])

    @torch.no_grad()
    def eval_loader(loader):
        return r2_last_global(model, loader, device)["r2_last"]

    # ---- 训练循环 + 早停 ----
    best_metric = -1e9
    best_state = None
    patience = cfg["train"]["early_stopping"]["patience"]
    bad = 0

    for ep in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{cfg['train']['epochs']}")
        for xb, yb, mb in pbar:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            xb = torch.nan_to_num(xb, nan=0.0, posinf=1e6, neginf=-1e6)
            yb = torch.nan_to_num(yb, nan=0.0)
            mb = torch.nan_to_num(mb, nan=0.0)

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg["train"]["amp"]):
                yh = model(xb)
                loss = masked_mse(yh, yb, mb)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=float(loss))

        # 验证
        val_r2 = eval_loader(inner_test_loader)
        logger.info(f"Epoch {ep} | inner_dev R2(last) = {val_r2:.6f}")

        # 调度器 (首轮跳过，避免 Warning)
        if scheduler is not None and ep > 1:
            if sch_cfg["name"] == "plateau":
                scheduler.step(val_r2)
            else:
                scheduler.step()

        # 早停逻辑
        if val_r2 > best_metric + cfg["train"]["early_stopping"]["min_delta"]:
            best_metric, bad = val_r2, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            ckpt_dir = (root / Path(cfg["paths"]["checkpoints_dir"])).resolve()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_dir / "best.pt")
        else:
            bad += 1
            if cfg["train"]["early_stopping"]["enabled"] and bad >= patience:
                logger.info("Early stopping triggered.")
                break

    # 保存 summary
    metrics_dir = (root / Path(cfg["paths"]["metrics_dir"])).resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json({"best_inner_dev_r2": best_metric},
              metrics_dir / "train_summary.json")
    logger.info(f"Best inner_dev R2(last): {best_metric:.6f} (checkpoint saved).")


if __name__ == "__main__":
    main()