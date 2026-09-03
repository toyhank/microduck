import os

robot_xml_path = 'microduck_rl/src/mjlab_microduck/robot/microduck/robot_walk.xml'
scene_xml_path = 'microduck_rl/src/mjlab_microduck/robot/microduck/scene_walk.xml'

# 1. Modify robot_walk.xml
content = open(robot_xml_path, 'r', encoding='utf-8').read()
if 'name="egocentric"' not in content:
    cam_str = '<camera name="egocentric" pos="0.01175 0 -0.0735" quat="0.707107 0 0.707107 -0" fovy="90"/>'
    content = content.replace('<site group="3" name="head_camera"', cam_str + '\n              <site group="3" name="head_camera"')
    open(robot_xml_path, 'w', encoding='utf-8').write(content)

# 2. Modify scene_walk.xml
content_scene = open(scene_xml_path, 'r', encoding='utf-8').read()
if 'name="target_block"' not in content_scene:
    env_str = '''
    <camera name="third_person" pos="0.5 0.5 0.5" xyaxes="-1 1 0 -0.5 -0.5 1" mode="trackcom" />
    <body name="target_block" pos="0.15 0 0.05">
        <freejoint/>
        <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1" mass="0.01"/>
    </body>
    '''
    content_scene = content_scene.replace('</worldbody>', env_str + '\n    </worldbody>')
    open(scene_xml_path, 'w', encoding='utf-8').write(content_scene)
