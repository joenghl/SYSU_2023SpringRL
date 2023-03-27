import gym
from gym.envs.classic_control import rendering

# 创建一个渲染器
viewer = rendering.Viewer(800, 600)

# 创建一个圆形对象
circle = rendering.make_circle(radius=10)

# 设置圆形对象的颜色（红色）
circle.set_color(0.3, 1, 0.3)

# 添加圆形对象到渲染器
viewer.add_geom(circle)

# 渲染场景
for _ in range(2000):
    viewer.render()
