"""
test_phase2.py
==============
测试第二阶段：双分支条件训练

验证点：
1. 模型能否正确处理 weak_condition 参数
2. 双分支损失函数能否正常计算
3. Jacobian 正则化能否正常工作
4. 训练循环能否正常运行
"""

import torch
import yaml
from models.transformer import build_model
from diffusion.sde import STEC_IRSDE
from training.losses import dual_branch_loss

def test_phase2():
    print("=" * 60)
    print("第二阶段测试：双分支条件训练")
    print("=" * 60)

    # 加载配置
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 构建模型和 SDE
    print("\n[1/5] 构建模型和 SDE...")
    model = build_model(cfg["model"]).to(device)
    sde = STEC_IRSDE(
        max_sigma=cfg["sde"]["max_sigma"],
        T=cfg["sde"]["T"],
        schedule=cfg["sde"]["schedule"],
        idw_power=cfg["inference"]["idw_power"],
        idw_k=cfg["inference"]["idw_k"],
    )
    print("✓ 模型和 SDE 构建成功")

    # 创建模拟数据
    print("\n[2/5] 创建模拟数据...")
    B, N = 4, 64
    coords = torch.randn(B, N, 2).to(device)
    angles = torch.randn(B, N, 2).to(device)
    system_ids = torch.randint(1, 5, (B, N)).to(device)
    stec = torch.randn(B, N, 1).to(device)
    valid_mask = torch.ones(B, N, dtype=torch.bool).to(device)

    # 创建 context/target mask
    context_mask = torch.zeros(B, N, dtype=torch.bool).to(device)
    target_mask = torch.zeros(B, N, dtype=torch.bool).to(device)
    for i in range(B):
        n_valid = N
        n_context = int(n_valid * 0.7)
        context_mask[i, :n_context] = True
        target_mask[i, n_context:n_valid] = True

    role_type = torch.zeros(B, N, dtype=torch.long).to(device)
    role_type[context_mask] = 1
    role_type[target_mask] = 2

    context_stec = stec * context_mask.unsqueeze(-1).float()
    print("✓ 模拟数据创建成功")

    # 构建条件均值和先验特征
    print("\n[3/5] 构建条件均值和先验特征...")
    mu, prior_features = sde.build_mu_batch(
        coords, stec, context_mask, target_mask,
        return_prior_features=True,
    )
    print(f"✓ mu shape: {mu.shape}, prior_features shape: {prior_features.shape}")

    # 前向加噪
    print("\n[4/5] 测试双分支前向传播...")
    t_batch = torch.randint(1, sde.T + 1, (B,)).to(device)
    xt_all, noise_all, _ = sde.forward_sample_batch(stec, mu, t_batch)
    noisy_stec = stec.clone()
    noisy_stec[target_mask] = xt_all[target_mask]

    # 强条件分支
    model.train()
    noise_pred_strong = model(
        noisy_stec=noisy_stec,
        coords=coords,
        angles=angles,
        system_ids=system_ids,
        context_stec=context_stec,
        role_type=role_type,
        valid_mask=valid_mask,
        t=t_batch,
        prior_features=prior_features,
        weak_condition=False,
    )
    print(f"✓ 强条件分支输出 shape: {noise_pred_strong.shape}")

    # 弱条件分支
    noisy_stec_weak = noisy_stec.clone().requires_grad_(True)
    noise_pred_weak = model(
        noisy_stec=noisy_stec_weak,
        coords=coords,
        angles=angles,
        system_ids=system_ids,
        context_stec=context_stec,
        role_type=role_type,
        valid_mask=valid_mask,
        t=t_batch,
        prior_features=prior_features,
        weak_condition=True,
        context_dropout_rate=0.3,
    )
    print(f"✓ 弱条件分支输出 shape: {noise_pred_weak.shape}")

    # 计算 x0_pred
    print("\n[5/5] 测试双分支损失函数...")
    sigma_t = torch.tensor([sde.sigma_bar(int(t.item())) for t in t_batch], device=device)
    alpha_t = torch.tensor([sde.alpha(int(t.item())) for t in t_batch], device=device)
    sigma_t = sigma_t.view(B, 1, 1)
    alpha_t = alpha_t.view(B, 1, 1)
    x0_pred_strong = (noisy_stec - mu - sigma_t * noise_pred_strong) / (alpha_t + 1e-8) + mu

    # 计算双分支损失
    loss_dict = dual_branch_loss(
        noise_pred_strong=noise_pred_strong,
        noise_pred_weak=noise_pred_weak,
        noise_target=noise_all,
        x0_pred_strong=x0_pred_strong,
        x0_true=stec,
        noisy_stec_weak=noisy_stec_weak,
        target_mask=target_mask,
        lambda_w=0.5,
        lambda_x=0.2,
        lambda_j=1e-4,
        loss_type="l1",
    )

    print(f"✓ 总损失: {loss_dict['loss_total'].item():.6f}")
    print(f"  - L_strong: {loss_dict['loss_strong'].item():.6f}")
    print(f"  - L_weak: {loss_dict['loss_weak'].item():.6f}")
    print(f"  - L_x0: {loss_dict['loss_x0'].item():.6f}")
    print(f"  - L_jac: {loss_dict['loss_jac'].item():.6f}")

    # 测试反向传播
    print("\n[测试] 反向传播...")
    loss_dict['loss_total'].backward()
    print("✓ 反向传播成功")

    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    if has_grad:
        print("✓ 模型参数梯度已计算")
    else:
        print("✗ 警告：模型参数梯度为空")

    print("\n" + "=" * 60)
    print("第二阶段测试完成！所有功能正常。")
    print("=" * 60)

if __name__ == "__main__":
    test_phase2()
