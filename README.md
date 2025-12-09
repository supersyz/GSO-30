# GSO-30 数据集使用说明

GSO-30 包含 30 个 GSO 对象，用于评估 3D-AIGC 方法。

我们在未公开论文中使用该数据集：
LSS3D：Learnable Spatial Shifting for Consistent and High-Quality 3D Generation from Single-Image（已提交 CVPR）

## 下载
- 文件：`GSO-30.zip`
- 网盘链接：https://pan.baidu.com/s/1G7S22KlOZPJp1AgCadTwGA
- 提取码：`w3jc`

## 校验
下载完成后校验 SHA-256：
```bash
shasum -a 256 GSO-30.zip
```
期望值：
```
5b364580ef19a8a2f482d4b929391588fc7d5f64bb8c1f341c70f2e9809c6ed7
```

## 预处理
解压后运行预处理脚本：
```bash
bash 1_gt_preprocess.sh
```

## 说明
- 数据仅用于学术研究与方法评估。
- 如在使用中遇到问题，欢迎在 Issues 反馈。
