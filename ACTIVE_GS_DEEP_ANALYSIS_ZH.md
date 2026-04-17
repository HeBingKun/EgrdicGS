# ActiveGS 仓库深度分析（中文）

仓库地址：[dmar-bonn/active-gs](https://github.com/dmar-bonn/active-gs)  
论文标题：*ActiveGS: Active Scene Reconstruction Using Gaussian Splatting*  
论文链接：[arXiv:2412.17769](https://arxiv.org/abs/2412.17769)

---

## 1. 先给结论：这个仓库到底在做什么？

ActiveGS 不是“先采完数据、再离线训练”的传统 3DGS 工程，而是一个**在线闭环系统**：

1. planner 依据当前地图，选出下一个最值得去的视角（NBV, next-best-view）。
2. simulator 在该视角生成 RGB-D 观测。
3. mapper 用新观测更新两张地图：
   - 一张是连续的 `GaussianMap`，负责重建和渲染。
   - 一张是离散的 `VoxelMap`，负责占据、frontier、ROI 和路径规划。
4. 更新后的 `GaussianMap` 再被 planner 用来“预测如果去某个候选视角，会看到多大不确定性/多少未知空间”。
5. 如此循环，直到时间预算耗尽。

一句话概括：**VoxelMap 负责“我能不能去”，GaussianMap 负责“我去了值不值”。**

---

## 2. 仓库的总控制流

### 2.1 入口：`main.py`

关键文件：`main.py:25-91`

按步骤看：

1. `main(cfg)` 读取 Hydra 配置。
2. 如果不是 debug 模式，就创建实验目录并保存 `exp_config.yaml`。
3. 初始化三大组件：
   - `mapping_agent = get_mapper(cfg, device)`
   - `simulator = get_simulator(cfg)`
   - `planner = get_planner(cfg, device)`
4. 如果打开 GUI，就建立 mapper/planner 与 GUI 的消息队列。
5. 把 `recorder / simulator / planner` 注入到 mapper。
6. 调用 `mapping_agent.run()`，真正开始在线重建任务。

这里有一个很重要的设计：**主程序并不自己维护“训练循环”或“推理循环”，真正的任务状态机在 `IncrementalMapper.run()` 里。**

---

## 3. 任务级主循环：规划 -> 采样 -> 训练 -> 占据更新

### 3.1 任务状态机：`mapping/mapper.py`

关键文件：`mapping/mapper.py:73-129`

这段代码是整个系统最重要的“外层循环”。

### 3.2 逐步骤注释

#### 步骤 A：初始化两张地图

位置：`mapping/mapper.py:42-45`

```python
self.gaussian_map = GaussianMap(self.cfg.gaussian_map, self.device)
self.voxel_map = VoxelMap(self.cfg.voxel_map, self.simulator.bbox, self.device)
```

含义：

- `GaussianMap` 是可训练的高斯显式场景表示。
- `VoxelMap` 是用于导航/路径规划/ROI 发现的离散占据图。

这正是论文的核心系统观：**连续重建表示 + 离散规划表示** 双图协同。

#### 步骤 B：planner 选路径与下一个关键帧

位置：`mapping/mapper.py:47-71`

```python
path = self.planner.plan(self.current_map, self.simulator, self.recorder)
dataframe = self.simulator.simulate(path[-1])
```

含义：

- planner 输出一条 camera path。
- 最后一个 pose 就是本轮 NBV。
- 这个 NBV 生成的 RGB-D 观测会作为新的 keyframe。

#### 步骤 C：更新 GaussianMap

位置：`mapping/mapper.py:100-101`

```python
self.gaussian_map.update(dataframe)
```

内部实际展开为：

1. `add_gaussians(dataframe)`：根据新观测补充新的高斯 primitive。
2. `train()`：对整张高斯地图做若干步增量优化。
3. `post_processing()`：更新 confidence，按周期 prune。

#### 步骤 D：更新 VoxelMap

位置：`mapping/mapper.py:103-104`

```python
self.voxel_map.update(dataframe)
```

作用：

- 用深度图更新 occupancy log-odds。
- 刷新 free / occupied / unexplored 的体素状态。
- 为下一轮 planner 的 frontier 和路径搜索提供依据。

#### 步骤 E：记录与可视化

位置：`mapping/mapper.py:109-125`

系统会把最新高斯地图、体素图、相机帧发给 GUI，并按时间间隔保存 map 与路径。

---

## 4. “训练流程”深拆：`GaussianMap`

> 这部分对应论文第 III-B、III-C 节，是仓库里最接近“在线训练框架”的核心。

关键文件：`mapping/gaussian_map.py`

---

### 4.1 `GaussianMap` 维护了哪些状态？

位置：`mapping/gaussian_map.py:17-60`

分两类：

#### 1. 可训练参数

- `_means`：高斯中心
- `_scales`：高斯尺度
- `_rotations`：高斯旋转
- `_opacities`：高斯不透明度
- `_harmonics`：颜色参数

#### 2. 非训练但对主动规划很关键的统计量

- `view_scores`
- `view_supports`
- `view_means`

这三项不是普通 3DGS 里的标准参数，它们是 **ActiveGS 论文创新的核心实现**，用于构造每个高斯的 confidence。

---

### 4.2 更新入口：`update()`

位置：`mapping/gaussian_map.py:62-64`

```python
def update(self, dataframe):
    self.add_gaussians(dataframe)
    self.train()
```

这说明 ActiveGS 每收到一个新关键帧，不是只做纯“densification”，而是：

1. 先补新高斯。
2. 再立刻对整图做短程增量训练。

这就是论文里强调的 **online incremental Gaussian mapping**。

---

### 4.3 高斯生成阶段：`add_gaussians()`

位置：`mapping/gaussian_map.py:294-468`

这是“从当前观测中生成新 primitive”的完整流程。

#### 步骤 1：从深度图恢复点云

位置：`mapping/gaussian_map.py:297-308`

代码块职责：

- 对深度图做平滑 `get_smooth_depth(...)`
- 按像素发射射线 `get_world_rays(...)`
- 用 `origin + direction * depth` 恢复世界坐标点云 `pcd`

对应论文第 III-B 节：把每个新 keyframe 变成候选 surfel / Gaussian。

#### 步骤 2：由深度估计法向

位置：`mapping/gaussian_map.py:315-336`

代码块职责：

- 调用 `depth2normal(...)` 从深度恢复法向。
- 把相机系法向变到世界系。
- 过滤不可见法向与异常法向。

为什么重要：

- ActiveGS 不是纯各向同性点高斯，而更接近**表面导向的 Gaussian surfel**。
- 法向决定初始旋转，进而决定高斯朝向。

#### 步骤 3：如果地图已初始化，先渲染当前地图做误差对比

位置：`mapping/gaussian_map.py:338-368`

代码块职责：

- 用当前已有高斯地图从本次视角渲染出 `rgb/depth/opacity/confidence/normal`。
- 生成 `global_render_results`，供后面筛选“哪里需要新高斯”。

这一步非常关键：**新高斯不是对每个深度像素都盲加，而是只在当前地图解释不好的地方补加。**

#### 步骤 4：初始化候选高斯属性

位置：`mapping/gaussian_map.py:370-381`

初始化逻辑：

- `means_new = pcd`
- `rotations_new, _ = normal2rotation(pcd_normals)`
- `scales_new[:, -1] -= 1e10`
- `opacities_new = 0`
- `harmonics_new[:, 0, :] = rgb`

解释：

- 均值直接来自深度反投影点。
- 旋转来自表面法向。
- 尺度的第三轴被压得极小，体现“薄片/表面高斯”的假设。
- opacity 的底层参数初始化为 0，对应 sigmoid 后大约 0.5。
- 颜色直接从当前像素赋值。

这正对应论文第 (2)(3)(4) 式的初始化思想。

#### 步骤 5：构造用于 confidence 的统计量占位

位置：`mapping/gaussian_map.py:378-381`

```python
view_scores_new = torch.zeros(...)
view_supports_new = torch.zeros(...)
view_means_new = torch.zeros(...)
```

这一步是普通 GS/MonoGS 里没有的，专门服务于 ActiveGS 的主动规划模块。

#### 步骤 6：筛选哪些像素真的要生成新高斯

位置：`mapping/gaussian_map.py:392-408` 与 `mapping/gaussian_map.py:470-489`

核心函数：`cal_mask(...)`

判断逻辑：

1. 如果当前地图颜色误差大，就补新高斯。
2. 如果当前像素渲染不透明度低，说明没被现有高斯充分解释，也补。
3. 如果预测深度比 GT 明显更远，也补。

实现代码：

```python
rgb_error = torch.mean((rgb_gt - rgb) ** 2, dim=1)
mask = rgb_error > self.error_thres
mask += opacity < 0.5
mask += (depth_gt.squeeze(0) - depth) < -0.05 * depth_gt.squeeze(0)
```

含义：

- 第一个条件对应“当前模型解释失败”。
- 第二个条件对应“当前表面没有高斯覆盖”。
- 第三个条件对应“当前地图深度落后于真实表面”，需要补表面。

#### 步骤 7：体素降采样，避免新高斯爆炸

位置：`mapping/gaussian_map.py:400-408`

这一步用 `voxel_downsample(...)` 对被选中的点做稀疏化，控制在线增量建图时高斯数量增长。

这也是论文能在线运行的重要工程手段。

#### 步骤 8：把选中的新高斯拼接进全局地图

位置：`mapping/gaussian_map.py:410-467`

做法是直接 `torch.cat` 到全局参数张量末尾，同时把这帧加入：

- `training_data`
- `training_performance`

也就是说，**每一帧既提供新几何，也进入未来增量训练缓存。**

---

### 4.4 增量训练阶段：`train()`

位置：`mapping/gaussian_map.py:66-130`

这部分是最标准的“训练循环”，但它是在线、小步、多次触发的。

#### 步骤 1：把当前参数包装成 `nn.Parameter`

位置：`mapping/gaussian_map.py:259-289`

`init_training()` 会把 `_means/_scales/_rotations/_opacities/_harmonics` 转成可优化参数，并用 Adam 配不同学习率。

特点：

- `mean/rotation` 学习率较小，避免几何大抖动。
- `opacity/scale` 学习率更大，便于快速适应新观测。

#### 步骤 2：构造训练帧采样器

位置：`mapping/gaussian_map.py:72-73`, `mapping/utils.py:190-228`

默认配置在 `config/mapper/incremental.yaml`：

- `batch_size = 8`
- `active_size = 3`
- `sampler_type = weighted`

`WeightedSampler` 的逻辑：

1. 最新的 `active_size` 帧一定进 batch。
2. 更早的帧按照 `training_performance` 权重采样。

这和论文“最近帧 + 历史难样本”的在线增量优化思想一致。

#### 步骤 3：渲染当前 batch

位置：`mapping/gaussian_map.py:80-104`

```python
GaussianRenderer(...).render_view_all(require_grad=True)
```

输出：

- `rgb_preds`
- `depth_preds`
- `normal_preds`
- `opacity_preds`
- `d2n_preds`

注意：这里的“推理”就是高斯前向渲染；在 ActiveGS 中，**训练、规划、评估都复用了同一个渲染器。**

#### 步骤 4：计算每帧表现，反哺采样权重

位置：`mapping/gaussian_map.py:109-111`, `132-139`

做法：

- 记录每帧的 RGB 和 depth 误差。
- 把误差写回 `training_performance[frame_ids]`。

意义：

- 后续 `WeightedSampler` 会优先抽“当前还没拟合好”的老帧。
- 这是一个很实用的在线 replay 机制。

#### 步骤 5：构造总损失

位置：`mapping/gaussian_map.py:113-124`

```python
total_loss = (
    rgb_loss
    + 0.8 * depth_loss
    + 0.1 * consistency_loss
    + 0.1 * normal_cons_loss
)
```

分别对应：

1. `rgb_loss`：颜色重建
2. `depth_loss`：深度重建
3. `consistency_loss`：渲染法向与 depth-to-normal 的一致性
4. `normal_cons_loss`：局部法向平滑正则

与论文公式对应关系：

- 论文式 (5) 给出的主体是 RGB + depth + normal consistency。
- 仓库实现比论文正文写得更“工程化”，多保留了 edge-aware normal TV 正则。

对应损失函数在：

- `mapping/utils.py:14-16`
- `mapping/utils.py:28-39`
- `mapping/utils.py:120-121`

#### 步骤 6：反向传播与更新

位置：`mapping/gaussian_map.py:125-127`

标准 Adam 更新，没有额外 densify/split/clone 机制；ActiveGS 主要通过“从新观测补 primitive”来做增量增长。

---

### 4.5 训练后处理：confidence 更新与 pruning

位置：`mapping/gaussian_map.py:141-246`

这是论文创新最强的一段实现。

#### 步骤 1：决定是否做全量 prune

位置：`mapping/gaussian_map.py:146-169`

- 每 `prune_interval` 帧做一次全训练视角可见性统计。
- 平时只用最新视角更新 confidence。

这是在线系统中的典型折中：**高频局部更新 + 低频全局清理**。

#### 步骤 2：重新渲染，统计每个高斯是否在视角中可见

位置：`mapping/gaussian_map.py:171-193`

注意这里：

```python
render_masks=(depth_gts > 0.0).float()
front_only=True
require_importance=True
```

作用：

- 只在真实观测有效区域内统计。
- 只考虑前表面贡献。
- 获取每个高斯在每视角中的 `count`。

#### 步骤 3：更新视角支持次数

位置：`mapping/gaussian_map.py:194-196`

```python
update_mask = counts[-1] >= 1.0
self.view_supports += update_mask.float()
```

含义：

- 某个高斯如果在最新视角中被真正看到，就认为它多获得了一次观察支持。

#### 步骤 4：更新 view distribution

位置：`mapping/gaussian_map.py:198-226`

这里实现了论文式 (6)(7)(8) 的思想。

核心量：

1. `view_directions`
   - 当前相机看向每个高斯的方向
2. `self.view_means`
   - 所有历史观测方向的均值向量
3. `self.view_scores`
   - 由“距离更近 + 与法向更一致”的视角累积出来的可见性质量分数

代码含义：

- `delta / self.view_supports` 是在线均值更新。
- `cosine_sim` 鼓励从“正面”观察高斯。
- `1 - distance_factor` 鼓励近距离观察。

也就是说，ActiveGS 的 confidence 不是“被看过几次”这么简单，而是：

**被多少个高质量视角、从多均匀的方向看过。**

#### 步骤 5：周期性 prune

位置：`mapping/gaussian_map.py:228-246`

策略：

1. 若某高斯在所有训练视角下都不可见，删。
2. 若 opacity 太低，也删。

这保证地图不会无限膨胀。

---

### 4.6 confidence 的最终形式

位置：`mapping/gaussian_map.py:551-565`

```python
view_var = self.view_means.norm(dim=-1)
view_variance_factor = torch.exp(1 - view_var)
confidences = torch.clamp(
    view_variance_factor * self.view_scores, min=0, max=1
)
```

解释：

- 如果一个高斯总是被从单一方向观察，那么 `view_means` 的模长会更接近 1。
- 如果它被从多方向均匀观察，均值向量会更“抵消”，模长更小。
- `exp(1 - ||mu||)` 会给多视角覆盖更高奖励。

所以 confidence 同时编码了：

1. 观测次数
2. 视角质量
3. 视角分布均匀性

这就是论文最关键的创新之一。

---

## 5. “推理流程”深拆：渲染器 `GaussianRenderer`

关键文件：`utils/operations.py:723-904`

这里的“推理”不是分类/检测式 inference，而是**给定相机位姿，前向渲染出高斯地图的 RGB/Depth/Normal/Confidence**。

### 5.1 初始化渲染器

位置：`utils/operations.py:724-779`

做的事情：

1. 读取高斯属性 `means/harmonics/opacities/confidences/scales/rotations`
2. 根据外参、内参构造 view/projection matrix
3. 根据分辨率生成 ray direction map
4. 准备 `render_masks`

这一步是训练、规划、评估共享的统一渲染前处理。

### 5.2 单视图前向：`render_view()`

位置：`utils/operations.py:790-826`

作用：

- 对第 `i` 个视角调用 CUDA rasterizer。
- 返回当前视角下的：
  - RGB
  - depth
  - normal
  - opacity
  - d2n
  - confidence
  - importance
  - count

### 5.3 多视图前向：`render_view_all()`

位置：`utils/operations.py:828-904`

作用：

- 对一批训练帧逐个渲染，堆成 batch 输出。
- 训练时就用它做多视图 supervision。

### 5.4 底层 CUDA rasterizer 输出了什么？

位置：`utils/operations.py:682-720`

底层 rasterizer 不只输出渲染颜色，还直接输出：

- `confidence`
- `importance`
- `count`

这意味着 ActiveGS 已经把主动规划所需的统计量直接嵌入到 GS 渲染过程中，而不是另起一个代价昂贵的体渲染/不确定性估计模块。

这也是它能比 NeRF/Fisher 信息类主动重建方法更快的重要原因。

---

## 6. 主动规划流程深拆：`PlanBase` + `Confidence`

> 这部分对应论文第 III-D、III-E 节。

---

### 6.1 `PlanBase.plan()` 的整体结构

关键文件：`planning/plan_base.py:41-129`

按执行顺序：

1. 更新可通行图 `voxel_map.update_graph(...)`
2. 生成 ROI 候选与随机候选
3. 计算每个候选视角的 utility
4. 用 A* 算从当前位置到候选点的路径长度
5. `utility - lambda * cost` 选 NBV
6. 把离散 waypoints 转成连续相机轨迹

这就是完整的“candidate-based NBV planning”。

---

### 6.2 ROI 是怎么来的？

#### 1. Frontier ROI

位置：`mapping/voxel_map.py:67-68`, `340-349`

逻辑：

- `frontier_mask = dilated(unexplored) & free_mask`
- 代表“靠近未知空间的自由体素”

#### 2. Low-confidence ROI

位置：`mapping/voxel_map.py:70-116`

逻辑：

1. 取所有高斯的 `confidence/opactiy/normal/mean`
2. 找出 `confidence < threshold 且 opacity > 0.7` 的高斯
3. 把这些高斯投票进对应 voxel
4. 如果某 voxel 里低置信高斯足够多，就把它标记成 ROI

意义：

- frontier 代表“还没看过的地方”
- low-confidence ROI 代表“虽然看过，但质量还差的地方”

这正是论文的探索-利用统一。

---

### 6.3 ROI 候选视角如何生成？

位置：`planning/plan_base.py:152-206`, `planning/utils.py:9-47`

逻辑：

1. 从 ROI 体素中心出发。
2. 在自由空间里找一个锥体区域内的候选相机位置：
   - 距离不能太近也不能太远
   - 观察方向要与 ROI 法向大致相对
3. 用 `rotation_from_z_batch(...)` 把“看向 ROI 的方向”转成相机旋转矩阵。

这一步的意义是：**候选视角不是完全随机，而是围绕“值得看”的区域定向采样。**

---

### 6.4 随机候选视角如何生成？

位置：`planning/plan_base.py:131-150`

逻辑：

- 只在当前位置半径 `radius` 内、且在 `free_mask_w_margin` 的体素里采样。
- 随机位置 + 随机朝向。

为什么还要随机候选：

- 防止 ROI 过强导致局部最优。
- 给 planner 保留一定全局探索能力。

---

### 6.5 候选视角 utility 如何计算？

关键文件：`planning/confidence.py:12-109`

这部分是论文最重要的主动规划创新之一。

#### 步骤 1：低分辨率渲染候选视角

位置：`planning/confidence.py:15-32`

```python
render_resolution = np.round(self.render_ratio * simulator.resolution)
renderer = GaussianRenderer(...)
```

这是非常关键的工程点：

- planner 不在原分辨率上评估候选，而是低分辨率快速预渲染。
- 因为 candidate 数量很多，这一步直接决定在线性能。

#### 步骤 2：计算 exploration utility

位置：`planning/confidence.py:69-86`

逻辑：

1. 用当前视角渲染的深度图构造 `depth_voxel`
2. 通过 `voxel_map.cal_visible_mask(...)` 得到该视角能看到哪些体素
3. 与 `unexplored_mask` 相交
4. 可见未知体素比例就是 exploration utility

公式上对应论文式 (9) 的第一项。

#### 步骤 3：计算 exploitation utility

位置：`planning/confidence.py:88-101`

逻辑：

1. 取渲染得到的 per-pixel confidence
2. 对无效区域和超出深度范围的区域，直接设为高置信，避免误判
3. 令 `uncertainty = 1 - confidence`
4. 再乘一个深度权重 `depth_surface / depth_range[1]`

实现上：

```python
dist_aware_uncertainty = (uncertainty) * depth_surface / depth_range[1]
exploit_util[i] = torch.mean(dist_aware_uncertainty)
```

解释：

- 论文里写的是优先看低 confidence 区域。
- 代码实现比论文文字更进一步：又引入了距离感知，使远处的低置信区域权重更大。

这可以理解为一个**实现层增强版 exploitation**。

#### 步骤 4：融合探索与利用

位置：`planning/confidence.py:105-108`

```python
utility = self.explore_weight * explore_util.cpu() + exploit_util.cpu()
```

注意：

- `explore_weight` 默认非常大（配置里是 `1000.0`）。
- 这说明实现里仍然明显偏向“先覆盖未知空间”，低置信利用项更像一个 tie-breaker / refinement。

---

### 6.6 路径代价如何并入 NBV 打分？

位置：`planning/plan_base.py:216-233`

代码：

```python
view_scores = view_utilities - self.path_length_factor * path_lengths
```

在这之前，utility 和 path length 都先做归一化。

含义：

- 不是单纯找“信息量最高”的视角。
- 而是在“价值高”与“代价低”之间做折中。

这对应论文式 (10)。

---

### 6.7 A* 路径规划

位置：`planning/utils.py:79-145`

做法：

- 在 `VoxelMap.graph.dense_graph` 上跑 A*。
- 图节点是自由体素。
- 边是 26 邻接体素连接。

所以 ActiveGS 的运动规划不是连续优化，而是：

1. 先在体素图上找离散路径
2. 再通过 `wp2path(...)` 转成连续相机轨迹

---

## 7. `VoxelMap` 为什么必不可少？

关键文件：`mapping/voxel_map.py`

如果只有 GaussianMap，其实很难完成在线主动探索，因为：

1. 你不知道哪里可通行。
2. 你不知道 frontier 在哪里。
3. 你没法高效跑路径规划。

因此 VoxelMap 起到三个作用：

### 7.1 占据更新

位置：`mapping/voxel_map.py:126-182`

特点：

- 用 log-odds 更新占据状态。
- `frustum_hit_mask` 表示被深度命中的体素。
- `frustum_pass_mask` 表示射线穿过但没有命中的体素。
- 距离越远，传感器更新权重越小。

### 7.2 Frontier 提取

位置：`mapping/voxel_map.py:340-349`

未知区域膨胀后与自由空间求交，即 frontier。

### 7.3 安全通行空间

位置：`mapping/voxel_map.py:328-338`

`free_mask_w_margin` 会对 occupied 做膨胀，给机器人留出安全边距。

这使得 planner 不会把相机采样到贴墙或穿模位置。

---

## 8. 论文创新点，与代码实现的一一对应

下面是最值得关注的“论文贡献 -> 仓库实现”映射。

### 创新 1：用 Gaussian Splatting 做主动重建的在线场景表示

论文含义：

- 不是把 GS 只当离线渲染器，而是把它直接放进在线闭环。

代码落点：

- `mapping/gaussian_map.py:62-64`
- `mapping/gaussian_map.py:66-130`
- `utils/operations.py:723-904`

我的理解：

- 这里最大的创新不只是“用了 GS”，而是**把 GS 前向渲染统一复用到训练、规划、评估三个阶段**。

### 创新 2：增量式 primitive spawning，而不是全量重建

论文含义：

- 新观测到来后，只在当前地图解释不好的位置补高斯。

代码落点：

- `mapping/gaussian_map.py:338-408`
- `mapping/gaussian_map.py:470-489`

关键思想：

- 通过 RGB error / opacity / depth discrepancy 筛掉“不需要再补”的像素。
- 从而把增量更新限制在局部。

### 创新 3：基于视角分布的 Gaussian confidence

论文含义：

- 一个区域“被看过”不等于“被高质量地、充分地重建”。
- confidence 应该同时考虑：
  - 观察次数
  - 观察距离
  - 与表面法向夹角
  - 视角分布是否均匀

代码落点：

- `mapping/gaussian_map.py:194-226`
- `mapping/gaussian_map.py:551-565`

这是整篇论文最有辨识度的创新。

### 创新 4：探索与利用统一的 candidate utility

论文含义：

- exploration：看更多未知区域
- exploitation：补低 confidence 区域

代码落点：

- `planning/confidence.py:69-108`
- `mapping/voxel_map.py:62-116`

我的理解：

- 这比传统 frontier-only 方法更合理，因为 frontier 无法处理“已看过但质量差”的表面。
- 这也比纯不确定性驱动更稳，因为它保留了空间覆盖需求。

### 创新 5：ROI-guided candidate sampling

论文含义：

- 候选视角不该完全随机，而应围绕 frontier 和 low-confidence 区域采样。

代码落点：

- `mapping/voxel_map.py:62-116`
- `planning/plan_base.py:152-206`
- `planning/utils.py:9-47`

价值：

- 降低候选空间规模
- 提高候选质量
- 让在线规划更可实时

---

## 9. 代码实现相对论文正文的几个“工程化增强/差异”

这部分很值得注意，因为你在读论文时不一定能直接看到。

### 9.1 exploitation 在代码里加入了距离加权

位置：`planning/confidence.py:95-101`

论文直观上强调低 confidence 区域；代码里进一步让远处低置信区域权重变大。

### 9.2 新高斯生成的 RGB 判据在实现里是 MSE 阈值

位置：`mapping/gaussian_map.py:482-485`

论文文字更像是“颜色误差超过阈值”；代码具体写成了：

```python
rgb_error = torch.mean((rgb_gt - rgb) ** 2, dim=1)
mask = rgb_error > self.error_thres
```

也就是实现上用的是均方误差形式。

### 9.3 训练损失比论文正文更细

位置：`mapping/gaussian_map.py:115-123`, `mapping/utils.py:28-39`

论文核心写了 normal consistency；代码里又加入了 edge-aware 的 normal TV 正则。

### 9.4 这套系统没有传统 3DGS 的 clone/split densification 流程

ActiveGS 的增长策略主要依赖：

1. 从新观测中补 primitive
2. 周期性 prune

而不是离线 3DGS 常见的大量 split/clone heuristics。

---

## 10. 如果你想顺着代码自己读，推荐顺序

建议按下面顺序：

1. `main.py`
2. `mapping/mapper.py`
3. `mapping/gaussian_map.py`
4. `utils/operations.py` 里的 `GaussianRenderer`
5. `mapping/voxel_map.py`
6. `planning/plan_base.py`
7. `planning/confidence.py`
8. `planning/utils.py`
9. `simulator/habitat_simulator.py`
10. `eval.py` 与 `utils/evaluation_tool.py`

这样最容易把“数据从哪来、参数怎么训、planner 怎么用地图”串起来。

---

## 11. 一张“脑内流程图”

```text
当前地图
  ├─ GaussianMap  -> 可渲染 RGB/Depth/Normal/Confidence
  └─ VoxelMap     -> 可查询 free/occ/frontier/path

planner.plan(...)
  ├─ VoxelMap.update_utility(...) 找 ROI
  ├─ 生成 ROI/random candidates
  ├─ GaussianRenderer 渲染每个 candidate
  ├─ 计算 exploration + exploitation utility
  ├─ A* 计算 travel cost
  └─ 选出 NBV

simulator.simulate(NBV)
  └─ 生成 RGB-D dataframe

GaussianMap.update(dataframe)
  ├─ add_gaussians(...) 在解释失败区域补新高斯
  ├─ train(...) 多视图小步增量优化
  └─ post_processing(...) 更新 confidence 与 prune

VoxelMap.update(dataframe)
  └─ 更新 occupancy / free / unexplored

进入下一轮
```

---

## 12. 我的最终理解：这篇论文真正的新意在哪里？

如果只看标题，你可能会觉得它只是“把 GS 用到 active reconstruction”。  
但从代码实现看，真正的创新不是单点，而是下面这套组合：

1. **表示层**：用可快速渲染的 GaussianMap 取代 NeRF 类隐式表示，保证在线性。
2. **质量建模层**：不是只看是否被观测，而是定义了 per-Gaussian confidence，并把视角分布编码进去。
3. **规划层**：同时兼顾 frontier exploration 和 low-confidence exploitation。
4. **工程层**：用 VoxelMap 处理安全、路径和 frontier，用 GaussianMap 处理重建质量，两者职责清晰分离。

换句话说，ActiveGS 的价值不只是“快”，而是它把：

- **重建质量估计**
- **视角选择**
- **实时性**

三件事在一个显式高斯框架里真正接起来了。

---

## 13. 额外观察

### 13.1 GUI 队列里有一个小疑点

位置：`main.py:65-66`

```python
planner.q_planner2gui = q_planner2gui
planner.q_gui2planner = q_planner2gui
```

第二行看起来更像应当是 `q_gui2planner`。  
不过当前 planner 代码并没有真正消费这个队列，所以对主算法链影响不大，更像一个 GUI 交互遗留问题。

### 13.2 仓库中的“推理”本质上就是渲染

在这个项目里，所谓 inference 最接近下面三类前向：

1. 训练时，对 batch 视角渲染预测图
2. 规划时，对 candidate 视角渲染 confidence/depth
3. 评估时，对测试视角渲染 RGB/depth

它们都复用 `GaussianRenderer`，只是输入姿态、分辨率、是否求梯度不同。

---

## 14. 你接下来如果还想继续深挖，最值得追的三个问题

1. `diff_gaussian_rasterization_2d` 扩展具体怎样把 confidence / importance / count 编进 rasterizer。
2. `normal2rotation`、`voxel_downsample`、`render_cuda_core` 的底层数学细节。
3. `mesh_generation.py` 如何把高斯地图再转成可评估 mesh。

如果你愿意，我下一步可以继续做两件事中的任意一个：

1. 给你补一份“按函数逐行中文注释版”，重点覆盖 `GaussianMap.train/add_gaussians/post_processing` 与 `Confidence.cal_utility`。
2. 继续深挖论文公式，把式 (2) 到式 (10) 和每一行代码做更严格的一对一对应表。
