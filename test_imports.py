"""
测试所有模块导入是否正常
"""
import sys
import os

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("测试模块导入...")
print("-" * 60)

try:
    print("1. 测试 data 模块...")
    from data import STECEpochDataset, build_train_val_datasets
    from data import collate_fn, generate_context_target_mask, build_dataloader
    print("   ✓ data 模块导入成功")
except Exception as e:
    print(f"   ✗ data 模块导入失败: {e}")
    sys.exit(1)

try:
    print("2. 测试 models 模块...")
    from models.transformer import build_model, STECDiffTransformer
    print("   ✓ models 模块导入成功")
except Exception as e:
    print(f"   ✗ models 模块导入失败: {e}")
    sys.exit(1)

try:
    print("3. 测试 diffusion 模块...")
    from diffusion.sde import STEC_IRSDE
    print("   ✓ diffusion 模块导入成功")
except Exception as e:
    print(f"   ✗ diffusion 模块导入失败: {e}")
    sys.exit(1)

try:
    print("4. 测试 training 模块...")
    from training.trainer import Trainer
    from training.losses import noise_prediction_loss, dual_branch_loss
    print("   ✓ training 模块导入成功")
except Exception as e:
    print(f"   ✗ training 模块导入失败: {e}")
    sys.exit(1)

try:
    print("5. 测试 inference 模块...")
    from inference.sampler import STECSampler
    print("   ✓ inference 模块导入成功")
except Exception as e:
    print(f"   ✗ inference 模块导入失败: {e}")
    sys.exit(1)

try:
    print("6. 测试 utils 模块...")
    from utils.normalizer import CoordNormalizer, STECNormalizer
    from utils.logger import get_logger
    print("   ✓ utils 模块导入成功")
except Exception as e:
    print(f"   ✗ utils 模块导入失败: {e}")
    sys.exit(1)

print("-" * 60)
print("✓ 所有模块导入测试通过！")
print("\n注意：如果运行 train.py 时遇到 DLL 错误，这是 PyTorch 环境问题，")
print("不是代码问题。请检查 PyTorch 安装或尝试重新安装。")
