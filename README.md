# Steretracker：在CLion中运行视觉与多信息融合定位

本项目面向 **CLion用户**。常规操作是在CLion中打开项目、选好Docker工具链和运行模式，然后点击绿色运行按钮；CLion会自动编译并启动C++程序`GPNP`。**日常使用无需启动run.py，也无需配置Windows的Conda Python解释器。**

支持两种主要模式：仅视觉定位，以及单目图像＋IMU＋雷达＋ESKF多信息融合。项目还保留双目纯视觉位姿/视差入口；双目多传感器融合与实时串口/CAN入口尚未接通。

## 1. 在CLion中一键启动

### 1.1 已配置好的电脑

1. 打开Docker Desktop，等待引擎就绪。
2. 在CLion中打开**本项目根目录**，即包含`CMakeLists.txt`的文件夹。例如当前独立目录为`C:\D\Codes\Steretracker_Runtime_20261116`。
3. 等待CMake加载完成；若提示重新加载，选择Reload CMake Project。
4. 在右上角选择 **Release-Docker**。
5. 选择 **GPNP_Visual**（仅视觉）或 **GPNP_Fused**（多信息融合）。
6. 点击绿色三角运行按钮，或使用`Shift+F10`。CLion先构建`GPNP`，再运行所选配置。

后续切换实验只需要切换运行项并再次点击运行。首次编译时间较长，之后只重新编译发生变化的部分。

| CLion运行项 | 程序参数 | 功能 |
|---|---|---|
| GPNP_Visual | config/mono_visual.json | 单目仅视觉；不读取IMU/雷达，不启用ESKF |
| GPNP_Fused | config/mono_fused.json | 单目图像、IMU法向、雷达高度及C＋CI＋ESKF融合 |

两份共享运行项位于[.run/GPNP_Visual.run.xml](.run/GPNP_Visual.run.xml)和[.run/GPNP_Fused.run.xml](.run/GPNP_Fused.run.xml)。

### 1.2 新电脑首次配置CLion

新电脑或从Git克隆后，本机的`.idea`设置不会随代码提交，需要先准备Docker镜像和CLion工具链。输入视频及传感器数据也需自行准备，见第5节。

**第一步：准备Docker环境。**

安装并启动Docker Desktop，准备已安装OpenCV、Eigen3、CMake、编译器及ONNX Runtime的Linux镜像。当前电脑使用`stereotracker_env:latest`；代码仓库不包含此镜像和外部运行库。其他电脑可使用依赖等价的镜像，并相应修改下面的工具链和ONNX Runtime路径。

**第二步：在CLion配置Docker工具链。**

进入“设置 → 构建、执行、部署 → 工具链（Toolchains）”，新增或选择Docker工具链：

| 字段 | 当前配置 |
|---|---|
| 名称 | Docker |
| 环境 | Docker，连接本机Docker Desktop |
| 镜像 | stereotracker_env:latest，或已准备好的等价镜像 |
| CMake、构建工具、C/C++编译器 | 使用容器内工具，等待CLion检测通过 |

本机ONNX模式使用Linux构建环境，不选择Visual Studio/MinGW工具链。该入口无需NPU；`rknn`预设用于具备对应运行库的RK3588环境，不用于普通电脑Docker实验。

**第三步：创建CMake构建配置。**

进入“设置 → 构建、执行、部署 → CMake”，创建或核对：

| 字段 | 值 |
|---|---|
| 配置名称 | Release-Docker |
| 启用 | 是 |
| 构建类型 | Release |
| 工具链 | Docker |
| 构建目录 | cmake-build-release-docker |
| CMake选项 | -DGPNP_WITH_RKNN=OFF -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-gpu-1.20.1 |

ONNX Runtime安装在其他位置时，替换上表中的目录；该目录是**容器内路径**。点击应用，重新加载CMake，确认配置成功。

项目也提供`onnx`/`rknn` CMake预设，但共享运行项当前绑定`Release-Docker`。首次配置按上表使用相同名称即可；不需要同时启用所有预设。

**第四步：核对运行配置。**

进入“运行 → 编辑配置（Edit Configurations）”。共享运行项未自动出现时，手动新增两个 **CMake应用程序**：

| 字段 | 仅视觉 | 多信息融合 |
|---|---|---|
| 名称 | GPNP_Visual | GPNP_Fused |
| 目标/可执行文件 | GPNP | GPNP |
| CMake构建配置 | Release-Docker | Release-Docker |
| 程序参数 | config/mono_visual.json | config/mono_fused.json |
| 工作目录 | 项目根目录（共享配置使用$PROJECT_DIR$） | 同左 |
| 环境变量 | QT_QPA_PLATFORM=offscreen | 同左 |
| 启动前任务 | Build | Build |

**程序参数只填写配置文件路径。** `-DGPNP_WITH_RKNN=OFF`属于CMake选项，不能填到这里；`--max-frames`属于辅助Python启动器，也不能当作GPNP的帧数控制参数。

### 1.3 先运行25帧，再运行完整实验

在CLion编辑器中打开将要运行的JSON配置，将以下字段设为25，保存后点击对应运行项：

```json
"input_system": {
  "max_frames": 25
}
```

这里展示的是需要修改的字段，不是替换整个JSON文件。仅视觉修改`mono_visual.json`，融合修改`mono_fused.json`。

确认输入、检测和输出正常后，将`max_frames`改为0读到视频结束，或改为10500明确限制帧数，再点击运行。现有数据从0-based源索引12000开始，`frame_step=1`；换自己的视频时必须同时核对起点、帧率及时间侧车。

首次应先运行一组，结束后再切换另一组，避免争用计算资源。常规运行使用绿色Run按钮；复现实验保持Release构建。

### 1.4 从哪里看结果

运行控制台显示处理状态，输出写入所选JSON的`output.directory`：

| 配置 | 默认输出目录 |
|---|---|
| mono_visual.json | results/mono_visual |
| mono_fused.json | results/mono_fused |

**CLion直接运行不会自动添加时间戳。** 每次需要保留结果时，先在CLion中修改`output.directory`，例如`results/visual_run01`、`results/fused_run01`，再点击运行，避免重复使用同一目录覆盖文件。

仅视觉主要查看`poses.csv`；融合另外查看`gravity_poses.jsonl`、`radar_poses.jsonl`和`eskf_states.csv`。各文件及坐标含义见第7节。

## 2. 环境与常见启动问题

| 项目 | 要求 |
|---|---|
| IDE | CLion，使用CMake应用程序运行项 |
| 实际执行系统 | Docker中的Linux，或依赖齐全的Linux/WSL/RK3588 Linux环境 |
| 编译 | CMake 3.22+、C++17编译器 |
| 库 | OpenCV（含videoio、video、calib3d等组件）、Eigen3、ONNX Runtime |
| RKNN可选后端 | RK3588环境的librknnrt；当前构建仍需ONNX Runtime |
| Python | 仅辅助脚本需要；CLion直接运行GPNP不需要配置Python解释器 |

ONNX Runtime安装目录应包含`include/onnxruntime_cxx_api.h`和`lib/libonnxruntime.so`。项目含POSIX接口，原生Windows编译器不能直接替代Linux工具链。

| 现象 | 检查方法 |
|---|---|
| 控制台命令以C:/Develop/Conda/.../python.exe开头，并提示不是Windows程序 | 当前选了Python运行项；切换到GPNP_Visual或GPNP_Fused |
| 提示打不开config/tracker_config.json | GPNP未收到配置参数；填写上述mono_visual.json或mono_fused.json路径 |
| 提示打不开-DGPNP_WITH_RKNN=OFF | 把CMake选项误填进程序参数；按第1.2节移回CMake配置 |
| 找不到视频、模型或标注 | 核对输入是否准备齐全、工作目录是否为项目根目录，以及配置中的相对路径 |
| 缺少ONNX Runtime | 核对容器内安装及ONNXRUNTIME_ROOT，不填写Windows安装路径 |
| CMake提示缓存来自其他目录 | 在CLion重置缓存并重新加载，或更换一个新的构建目录；不要沿用搬迁前路径的缓存 |
| 右上角没有Release-Docker | 在CMake设置中按第1.2节创建并启用；重新加载项目 |

CLion构建目录使用`cmake-build-release-docker`，辅助脚本使用`build-onnx`，两者分开，避免不同容器映射路径造成缓存冲突。

## 3. 仅视觉模式

### 3.1 处理流程

```text
视频帧 / 图像
  → YOLO检测靶板候选ROI
  → 候选区域的几何、外观及身份验证
  → 按可见区域和尺度选择BC、TT或DualROI
  → 提取二维点并与模板三维点编号对应
  → 多候选单目PnP
  → 重投影、物理先验及连续性验收
  → 输出A：纯视觉位姿
```

- **BC（BinaryCorner）**：从轮廓提取角点并与模板编号对应，适用于具备足够像素跨度的目标。
- **TT（TinyTarget）**：为小目标提取四点；BC失败时也可进入TT回退。
- **DualROI**：在同一张图中使用外部class0与内部class1区域，尝试联合求解及class1回退。它不是双目相机。
- **PnP**：使用ITERATIVE、EPnP/RANSAC、平面IPPE及历史初值等候选，并执行相应验收；不是仅调用一次EPnP。

单目实际尺度来自靶板物理模型，相机K和靶板尺寸都会影响位置。当前780mm是已有数据的工作模型，不应直接当作任意场景的实测尺寸。

### 3.2 输入和开关

仅视觉模式需要图像、相机内参、检测模型、靶板模板和标注点。提供的启动配置使用离线视频；底层另有图片、图片序列和摄像头输入实现。

`mono_visual.json`关闭以下开关：

```json
{
  "gravity_pose": { "enabled": false },
  "radar_pose": { "enabled": false },
  "eskf": { "enabled": false }
}
```

`input_system.imu.enabled`与`input_system.altimeter.enabled`也均关闭，因此该入口不读取IMU或雷达，不产生融合状态。

## 4. 多信息融合模式

### 4.1 各分支关系

```text
图像 → 候选ROI与二维/三维对应点
          ├─ 纯视觉PnP                         → A
          ├─ 对应点＋IMU有向法向先验            → B
          └─ 对应点＋IMU法向先验＋雷达高度      → C

帧间全频IMU → ESKF状态传播
帧间独立雷达 → 高度更新
C条件位姿后验 → 门控与CI合入 → ESKF连续状态
```

A、B、C独立保存。B和C复用已经形成的视觉观测，不是逐次覆盖A的R/t；C也不是简单把A或B的`t.z`替换为雷达读数。

| 分支 | 使用信息 | 主要作用 |
|---|---|---|
| A | 图像特征与靶板模型 | 保留纯视觉参考 |
| B | 视觉对应点＋IMU法向 | 约束平面姿态的不稳定方向 |
| C | 视觉对应点＋IMU法向＋雷达高度 | 联合约束姿态与板面高度 |
| ESKF | 全频IMU、独立雷达、C条件后验 | 传播状态并按条件融合观测 |

B使用日志中已解算的Roll/Pitch，通过安装旋转转换为相机系有向板面法向，并施加软约束。这里不能仅凭原始加速度单样本就认定姿态可靠。

C联合优化图像重投影、法向及高度残差；高度关系为`R.col(2).dot(t_m)`，其中`t_m`使用米。当前雷达按高度类观测使用，不额外乘cos；安装偏移和高度偏置需按实际设备核对。

### 4.2 时间对齐和ESKF

程序用视频侧车的完整日期时间关联IMU/雷达，查找不晚于曝光对应时刻的历史样本，执行年龄和有效性检查。视频帧号/fps提供逻辑时钟，不能代替不同设备之间的实际时间对齐。

启用`eskf.pose_source=imu_radar_pnp`时，程序收集两帧之间的全频IMU/雷达事件并按时间顺序处理，而不是只把每帧一个IMU样本用于积分。

C已经使用了IMU和雷达，与滤波状态共享信息，所以通过**CI（Covariance Intersection，协方差交集）**入口合入，避免按相互独立观测直接重复叠加信息。门控或CI选择可能保留先验；C有输出不等于该帧一定应用了滤波更新。

缺少有效视觉观测时，已初始化的滤波器仍可传播，但需要同时检查状态质量，不能把连续输出视为始终有可靠视觉定位。

### 4.3 初始化与必要条件

当前默认配置：

- `mode=normal`、`mono_mode=true`，同步离线视频，关闭线程采集。
- `gravity_pose`、`radar_pose`、IMU文件源及雷达文件源同时启用。
- `vision_output.mode=standalone`、`eskf.pose_source=imu_radar_pnp`、`eskf.threaded=false`。
- IMU文件最早5秒保留合格的静止前缀，用于估计陀螺偏置。不能把视频截取起点自动当作静止。
- 通过五个一致C观测后初始化；默认`initialization_source=conditioned_pose`，不读取RTK初始位置。
- 初始XY由条件位姿转换得到；设置地面雷达参考后，初始Z使用当前雷达减该参考，并对后续输入作相应固定平移。

当前`initialization_ground_radar_height_m=0.6`来自已有数据的地面参考，不是通用值。`imu_to_camera_rotation`与`eskf.R_imu_cam`必须一致；`R_template_world`定义模型到融合世界系的方向，不能未经标定就称为精确ENU外参。

## 5. 准备数据

### 5.1 视频和时间侧车

`input_system.image`中的主要字段：

| 字段 | 含义 |
|---|---|
| video_path | OpenCV/FFmpeg能解码的视频文件，扩展名可以是.bin |
| timestamp_path | 按解码显示顺序逐帧对应的时间侧车 |
| video_fps | 视频帧率，当前25Hz |
| start_frame | 原视频起点，0-based |
| frame_step | 取帧步长，当前1 |
| seek_mode | sequential、auto或indexed |
| indexed_video_path | 可选快速定位缓存；还需同名.index.json |

时间侧车每行格式如下，编号连续且从1开始：

```text
1_2026_01_01_12_00_00_000
2_2026_01_01_12_00_00_040
```

`auto`模式无法使用有效缓存时会回退顺序解码；`indexed`要求缓存验证通过。换视频时不能沿用其他视频的缓存或侧车。

### 5.2 IMU文件

当前`flight_csv`入口要求9列，顺序固定。可包含以`timestamp,`或`时间戳,`开头的表头。下面仅示意格式：

```csv
timestamp,gx,gy,gz,ax_g,ay_g,az_g,roll_deg,pitch_deg
2026_01_01_12_00_00_000,0,0,0,0,0,1,0,0
```

| 列 | 单位与含义 |
|---|---|
| timestamp | YYYY_MM_DD_HH_MM_SS_mmm |
| gx、gy、gz | 陀螺角速度，度/秒；内部转为弧度/秒 |
| ax_g、ay_g、az_g | 加速度，以g为单位；内部乘9.81 |
| roll_deg、pitch_deg | 已解算倾角，度；按所配置轴序和安装关系解释 |

当前配置显式使用`gyro_unit=deg_s`、`acc_unit=g`、`normal_source=logged_tilt`、`tilt_order=roll_pitch`。日期时间不得倒退；`duplicate_policy=mean`时同毫秒记录取均值。实际融合必须提供连续完整采样及静止前缀，单行示例不能用于运行。

`attitude_valid_from`必须反映当前日志中可接受的姿态时段。现有倾角轴序、滤波延迟与粗安装外参仍应随新设备核验，不能机械套用。

### 5.3 雷达文件

CSV必须有表头；通过`timestamp_column`与`height_column`选择时间和高度列，默认分别为`aligned_time`、`height_m`：

```csv
aligned_time,height_m
2026-01-01 12:00:00.000,0.60
```

高度单位米。当前解析也接受下划线形式的毫秒日期时间；额外列允许保留，但各行列数须一致。程序检查范围、时序、重复时间及突变；重复时间的高度取均值。

IMU、雷达和视频侧车应事先映射到同一时间基准。现有对齐文件已完成时间平移，当前额外偏移为0，不应再重复减去旧偏移。

### 5.4 模型与标注

运行资源包括：

- `data/clean_target2.png`：完整靶板模板。
- `scripts/class0_points.txt`、`scripts/class1_points.txt`：两类有序角点及模板图像说明。
- `strategies.board_template.template_real_width_mm/height_mm`：完整模板的物理宽高。
- `camera.fx/fy/cx/cy`：与输入通道、分辨率及裁剪方式匹配的相机内参。
- `yolo_onnx/`中的检测权重。

统一模型由完整模板、两份标注及物理尺寸生成。更换模板或尺寸时同时核对标注图像尺寸、点编号和物理坐标；不要另行加入与统一模型冲突的BC比例或TT独立边长。

## 6. 在CLion中切换配置与替换数据

直接在CLion项目树中打开`config`目录，编辑对应JSON后保存，再运行相应的CMake运行项。替换数据时按第5节更新图像、IMU、雷达路径、K、模型尺寸及安装关系；不要只改视频路径就沿用全部旧标定。

| 需要改变的内容 | 修改位置 |
|---|---|
| 实验模式 | 右上角GPNP_Visual / GPNP_Fused |
| 处理帧数 | 配置的input_system.max_frames |
| 起点、步长、帧率和视频 | input_system.image |
| 相机内参 | camera |
| 靶板物理尺寸 | strategies.board_template |
| 传感器文件及单位、轴序、安装关系 | input_system.imu / input_system.altimeter |
| 融合初始化和噪声 | eskf |
| 结果目录及导出 | output |

所有相对路径均以项目根目录为基准。正式连续处理使用`mode=normal`；`mode=debug`会读取单张调试图，不启动完整视频融合。

当前纯视觉与融合配置分别控制相关传感器和求解开关；不建议在两者之间只零散切换一个enabled。初次使用先保持现有参数，确认流程后再按设备和实验设计修改。

## 7. 输出和坐标

CLion直接运行时，输出写入JSON指定的`output.directory`。需要留存本次参数时，应自行复制所用JSON到结果目录。只有辅助启动器`run.py`会自动创建`results/<模式>_<时间戳>/`并保存`config_used.json`。主要输出如下：

| 文件 | 含义 |
|---|---|
| poses.csv | A分支的逐帧视觉位姿及成功/失败标记；不是最终融合状态 |
| feature_points.csv | 像素点、编号及使用情况，开启导出时产生 |
| input_frames.csv | 处理帧、原视频源索引、时间及失败信息 |
| gravity_poses.jsonl | B分支结果，仅启用法向辅助时产生 |
| radar_poses.jsonl | C分支结果，仅启用雷达约束时产生 |
| imu_alignment.csv、radar_alignment.csv | 帧级对齐、样本年龄及有效性 |
| imu.csv、radar.csv | 条件融合路径实际处理的传感器事件 |
| eskf_states.csv | ESKF状态、质量、更新状态及协方差相关量 |
| eskf_events.csv | IMU传播、雷达更新、初始化和条件位姿处理决定 |
| tracking_log.txt | 跟踪日志，开启log_file时产生 |
| target_model.json | 实际使用的模型定义 |

A/B/C的R、t表示模型点到相机系的变换：`X_camera = R * X_model + t`，平移使用毫米。相机在模型系的位置为`C = -R.transpose() * t`，不能直接把`t`当作世界坐标中的相机位置。

融合输出的位置和速度分别使用米、米/秒，位置包含`R_template_world`转换与初始化平移。模型公共坐标原点为class0白色轮廓包围框中心；+X向右、+Y向下、+Z朝板背面。比较不同分支前必须统一参考系、单位和初始参考。

读取`eskf_states.csv`时同时检查`quality`与`status`：

- `uninitialized`：尚未完成初始化。
- `normal`、`degraded`、`stale`：状态质量分级，不能只按行数统计可靠定位。
- `ci_update`：本次应用了条件位姿CI更新。
- `prior_preferred`：保留先验；不能当作一次新的C位姿更新。
- 其他拒绝或缺测状态应结合事件日志解释。

没有输出的帧应单列，不赋零；输出数增加本身不能证明定位精度提升。

## 8. 在CLion中运行双目纯视觉

在“运行 → 编辑配置”中复制GPNP_Visual，保留目标GPNP、Release-Docker、项目根目录和Build任务，修改名称与程序参数：

| 名称示例 | 程序参数 | 功能 |
|---|---|---|
| GPNP_Stereo | config/stereo_sample_pose.json | 双目位姿 |
| GPNP_Disparity | config/stereo_sample_disparity.json | 仅匹配/视差 |

再选择对应运行项，点击绿色运行按钮。短运行同样在JSON中设置`input_system.max_frames=25`。

左右图平铺于同一目录，文件后缀严格配对：

```text
data/my_stereo/
  left_000001.png
  right_000001.png
  left_000002.png
  right_000002.png
```

修改`directory_path`、`left_pattern`、`right_pattern`以及K、基线、模型尺寸。当前主程序按共用K、左右旋转为单位阵、X向基线的模型处理，应使用与该几何关系匹配的左右图。

双目可使用传统两侧提取，或左侧特征＋右图LK匹配前端。位姿配置使用`disparity_only=false`，仅视差配置使用true。部分位姿路径存在左目PnP回退，应结合实际输出来源判断。

默认`data/stereo_crop`是裁剪/平移合成样例，当前400mm模型和继承内参未针对它独立标定，不能作为真实双目精度证明。`sequence`和`camera`会复制左图作为右图，不能用于真实双目采集；离线视频入口当前只允许单目。

## 9. 目录与版本管理

```text
main.cpp                  主程序
include/、src/            检测、特征、匹配、求解、跟踪、输入和融合
eskf/                     ESKF核心
config/                   四套运行配置
scripts/                  必需的class0/class1模型标注
third_party/              RKNN接口头文件与原版权声明
yolo_onnx/                检测模型
data/                     模板资源及本地输入
.run/                     CLion共享运行项
run.py、run.sh            可选命令行入口
START_HERE.md             CLion操作速查
results/                  本地输出，不提交
```

Git保存源码、正式配置、共享运行项、辅助启动入口、检测模型及靶板模板。视频、IMU/雷达原始材料、双目输入、定位缓存、构建产物、实验结果、材料清单及本次结果核验脚本均不提交。克隆代码后需自行准备输入，或使用当前电脑上保留的本地材料。

本目录不构建测试/对比脚手架，但主程序内部仍保留原Debug及模拟源实现；提供的配置不启用模拟传感器。对新数据仍需独立核对时间同步、相机/IMU外参、靶板尺寸、雷达高度语义和参考系。

## 10. 辅助方式：命令行与双击脚本

以下方式用于离开IDE后的批处理，不是CLion常规使用的必经步骤。

- Windows：打开Docker Desktop后双击`start_visual.cmd`或`start_fusion.cmd`，自动使用当前目录挂载并创建时间戳结果目录。
- PowerShell：在本目录执行`.\run_docker.cmd mono --max-frames 25`或`.\run_docker.cmd mono-fused --max-frames 25`。
- Linux/容器内：执行`python3 run.py mono`或`python3 run.py mono-fused`。

`run.py`负责选配置、检查输入、构建GPNP和创建时间戳结果目录；不是核心算法。它的`--max-frames`、`--backend`、`--jobs`和`--check-only`均属于Python启动器参数。CLion直接运行GPNP时，通过JSON控制帧数、输入和输出。

辅助脚本需要Python；Windows的Conda Python不能直接执行Linux算法，请使用Docker入口。更多辅助说明见[START_HERE.md](START_HERE.md)。
