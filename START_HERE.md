# CLion一键启动速查

完整使用说明以[README.md](README.md)为准。常规使用在CLion中直接运行C++目标GPNP。

## 日常操作

1. 打开Docker Desktop，等待引擎就绪。
2. 在CLion打开本项目，等待CMake加载成功。
3. 右上角选择Release-Docker。
4. 选择GPNP_Visual（仅视觉）或GPNP_Fused（多信息融合）。
5. 点击绿色运行按钮，或按Shift+F10。

无需点击run.py，也无需选择Windows Conda Python。CLion自动执行构建，再使用所选JSON运行GPNP。

## 首次配置

若没有Release-Docker，在“设置 → 构建、执行、部署 → CMake”创建：

- 配置名称：Release-Docker。
- 工具链：Docker，镜像使用stereotracker_env:latest或等价依赖镜像。
- 构建类型：Release。
- 构建目录：cmake-build-release-docker。
- CMake选项：-DGPNP_WITH_RKNN=OFF -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-gpu-1.20.1。

上述ONNX Runtime目录是当前镜像内路径，其他环境按实际安装修改。共享运行项在.run目录；未自动出现时按README第1.2节手动建立CMake应用程序。

| 运行项 | 目标 | 程序参数 |
|---|---|---|
| GPNP_Visual | GPNP | config/mono_visual.json |
| GPNP_Fused | GPNP | config/mono_fused.json |

两者均选择Release-Docker，工作目录为项目根目录，环境变量QT_QPA_PLATFORM=offscreen，运行前任务为Build。不要把-D选项或Python启动参数填到GPNP程序参数里。

## 25帧检查和完整运行

在CLion中编辑对应JSON，将input_system.max_frames设为25，保存后点击运行。检查通过后设为0读到结束，或10500明确限制帧数，再运行。换数据时同步核对start_frame、视频帧率、时间侧车、内参和靶板模型。

CLion直接运行使用JSON的output.directory，不自动增加时间戳。需要保留多次结果时，每次先改成不同目录，例如results/visual_run01和results/fused_run01。

## 查看输出

- A纯视觉：poses.csv。
- B视觉＋IMU法向：gravity_poses.jsonl。
- C视觉＋IMU＋雷达：radar_poses.jsonl。
- 融合状态：eskf_states.csv，结合quality/status阅读。

## 可选的非IDE方式

需要脱离CLion批处理时，打开Docker Desktop后可双击start_visual.cmd或start_fusion.cmd。它们使用独立容器和时间戳结果目录；也可在PowerShell执行run_docker.cmd。这些脚本是辅助入口，CLion日常操作不依赖它们。
