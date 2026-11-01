"""仿真观测相机：一台第三人称固定相机 + 一台腕部相机。

只在 `capture()` 中被调用，不推进物理，也不改变任何物体状态，因此不会影响
专家轨迹的可复现性。渲染使用 PyBullet 的 TINY_RENDERER，可完全无窗口运行。
"""
import numpy as np
import pybullet as p

from .config import IMAGE_SIZE

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def _downsample(rgb, out_size):
    """把 (H,H,3) 渲染结果缩到目标尺寸；有 PIL 时用 LANCZOS，保证下采样平滑。"""
    h, w = rgb.shape[:2]
    if (w, h) == (out_size[1], out_size[0]):
        return rgb
    if Image is None:
        step = max(1, w // out_size[1])
        return rgb[::step, ::step][: out_size[0], : out_size[1]]
    image = Image.fromarray(rgb).resize((out_size[1], out_size[0]), Image.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def _render(sim, view, proj, out_size, supersample=2):
    width, height = out_size[1] * supersample, out_size[0] * supersample
    pixels = p.getCameraImage(
        width, height, view, proj,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_NO_SEGMENTATION_MASK,
        physicsClientId=sim.client,
    )[2]
    rgb = np.asarray(pixels, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    return _downsample(rgb, out_size)


class CameraRig:
    """按 config.CAMERAS 的顺序采集多路 RGB，返回 {名称: uint8 (H,W,3)}。"""

    def __init__(self, sim, cameras=None, image_size=IMAGE_SIZE, supersample=2):
        from .config import CAMERAS

        self.sim = sim
        self.image_size = tuple(image_size)
        self.supersample = int(supersample)
        self.cameras = dict(cameras or CAMERAS)
        self._static = {}
        for name, cfg in self.cameras.items():
            if cfg['type'] == 'fixed':
                view = p.computeViewMatrixFromYawPitchRoll(
                    list(cfg['target']), cfg['distance'], cfg['yaw'], cfg['pitch'], 0, 2)
                proj = p.computeProjectionMatrixFOV(
                    cfg['fov'], self.image_size[1] / self.image_size[0], cfg['near'], cfg['far'])
                self._static[name] = (view, proj)
            elif cfg['type'] != 'wrist':
                raise ValueError(f'unknown camera type: {cfg["type"]}')

    def _wrist_matrices(self, cfg):
        state = p.getLinkState(self.sim.robot, cfg['link'], computeForwardKinematics=True,
                               physicsClientId=self.sim.client)
        position = np.asarray(state[4], dtype=float)
        quat = np.asarray(state[5], dtype=float)
        rotation = np.asarray(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
        eye = position + rotation @ np.asarray(cfg['offset'], dtype=float)
        approach = rotation[:, 2]
        if cfg.get('forward') is not None:
            # ---- 固定外参（实验 4.0 v3 起）：相机相对安装 link 的 SE(3) 恒定 ----
            # 视线方向与 up 都在 link 局部坐标系里给定，再经正运动学转到世界系。
            # 因此手指开合、EE 绕自身旋转都不会改变“相机相对 link 的姿态”，
            # 也绝不使用物体真值或任务阶段去瞄准。
            forward_local = np.asarray(cfg['forward'], dtype=float)
            up_local = np.asarray(cfg.get('up_local', (0.0, 1.0, 0.0)), dtype=float)
            forward_local = forward_local / np.linalg.norm(forward_local)
            target = eye + rotation @ forward_local * float(cfg.get('look_ahead', 0.3))
            up = rotation @ (up_local / np.linalg.norm(up_local))
        elif cfg.get('aim') == 'grasp':
            # 历史实现（实验 1–3）：朝向依赖两指尖中点 => 姿态随手指开合轻微变化。
            # 保留给旧配置，实验 4.0 的新配置不再使用这条分支。
            tips = [np.asarray(p.getLinkState(self.sim.robot, link, computeForwardKinematics=True,
                                              physicsClientId=self.sim.client)[4], dtype=float)
                    for link in (9, 10)]
            target = 0.5 * (tips[0] + tips[1]) + approach * float(cfg.get('look_ahead', 0.0))
            target[2] -= float(cfg.get('target_down', 0.0))
            up = np.array([0.0, 0.0, 1.0])
        elif cfg.get('aim') == 'ee':
            state_ee = p.getLinkState(self.sim.robot, self.sim.ee, computeForwardKinematics=True,
                                      physicsClientId=self.sim.client)
            target = np.asarray(state_ee[4], dtype=float)
            up = np.array([0.0, 0.0, 1.0])
        else:
            target = eye + rotation[:, 2] * cfg['look_ahead']
            up = rotation[:, 1]
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        proj = p.computeProjectionMatrixFOV(
            cfg['fov'], self.image_size[1] / self.image_size[0], cfg['near'], cfg['far'])
        return view, proj

    def wrist_extrinsics(self, cfg):
        """返回相机相对安装 link 的固定外参 (R_link_cam, t_link_cam, R_world_link)。

        只依赖 link 位姿与配置，不依赖手指/物体，因此可直接用来验证“刚性安装”。
        """
        state = p.getLinkState(self.sim.robot, cfg['link'], computeForwardKinematics=True,
                               physicsClientId=self.sim.client)
        position = np.asarray(state[4], dtype=float)
        quat = np.asarray(state[5], dtype=float)
        rotation = np.asarray(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
        eye = position + rotation @ np.asarray(cfg['offset'], dtype=float)
        if cfg.get('forward') is not None:
            forward_local = np.asarray(cfg['forward'], dtype=float)
            forward_local = forward_local / np.linalg.norm(forward_local)
            up_local = np.asarray(cfg.get('up_local', (0.0, 1.0, 0.0)), dtype=float)
            up_local = up_local / np.linalg.norm(up_local)
            target = eye + rotation @ forward_local
            up = rotation @ up_local
        else:
            view, _ = self._wrist_matrices(cfg)
            V = np.asarray(view, dtype=float).reshape(4, 4).T
            R_cw = V[:3, :3]
            R_wc = R_cw.T
            eye = -R_wc @ V[:3, 3]
            target = eye - R_wc[2, :]
            up = R_wc[1, :]
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        V = np.asarray(view, dtype=float).reshape(4, 4).T
        R_cw = V[:3, :3]          # camera <- world
        R_wc = R_cw.T             # world <- camera
        R_lw = rotation           # world <- link
        R_lc = R_lw.T @ R_wc      # camera <- link（固定外参的旋转部分）
        t_lc = R_lw.T @ (eye - position)
        return {'R_link_camera': R_lc, 't_link_camera': t_lc, 'R_world_link': R_lw,
                'eye_world': eye, 'link_position': position}

    def capture(self):
        frames = {}
        for name, cfg in self.cameras.items():
            view, proj = self._static[name] if cfg['type'] == 'fixed' else self._wrist_matrices(cfg)
            frames[name] = _render(self.sim, view, proj, self.image_size, self.supersample)
        return frames
