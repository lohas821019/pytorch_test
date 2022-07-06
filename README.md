# 先行準備
<br>

[首先去pytorch官網安裝](https://pytorch.org/get-started/locally/)

這邊的範例是使用cuda 11.3 的GPU版本

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

<br>

若已安裝，可使用下列指令確認版本，若沒就直接去[Nvidia官方下載](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)

```
nvcc --version
```

[很棒的參考資料](https://docs.microsoft.com/zh-tw/windows/ai/windows-ml/tutorials/pytorch-data)