# model_renderer
Two wrappers to blender for generating object images at various orientations, originally based off of [RenderForCNN](https://github.com/ShapeNet/RenderForCNN)'s renderer. Depends on [quat_math](https://github.com/r-pad/quat_math).

Blender as a Python Module
-----
This script will clone blender to where ever its run and installs bpy to the current virtual environment. Requires python3.6.
This script assumes you are in a virtual environment. If you want to install without a virtual environment, change the PYENV to the relavent locations.
```
sudo apt-get install gawk cmake cmake-curses-gui build-essential libjpeg-dev libpng-dev libtiff-dev git libfreetype6-dev libx11-dev flex bison libtbb-dev libxxf86vm-dev libxcursor-dev libxi-dev wget libsqlite3-dev libxrandr-dev libxinerama-dev libbz2-dev libncurses5-dev libssl-dev liblzma-dev libreadline-dev libopenal-dev libglew-dev yasm libtheora-dev libvorbis-dev libogg-dev libsdl1.2-dev libfftw3-dev patch bzip2 libxml2-dev libtinyxml-dev libjemalloc-dev libjpeg-dev libopenimageio-dev libopencolorio-dev libboost-all-dev libopenexr-dev libsndfile1-dev libx264-dev autotools-dev libtool m4 automake cmake libblkid-dev e2fslibs-dev libaudit-dev libavformat-dev ffmpeg libavdevice-dev libswscale-dev libopenjp2-7-dev libalut-dev libalut0 libmp3lame-dev libspnav-dev libspnav0 spacenavd

. venv/bin/activate
pip install numpy requests
./install_bpy.bash
python -c "import bpy ; bpy.ops.render.render(write_still=True)"
```

If you get the error
```
error: the type ‘const OpenImageIO::v1_7::TypeDesc’ of constexpr variable ‘ccl::TypeFloat2’ is not literal
```
change line 31 of blender-git/blender/intern/cycles/util/util_param.h to 
```
static const/*expr*/ TypeDesc TypeFloat2(TypeDesc::FLOAT, TypeDesc::VEC2);
```

BpyRenderer
-----
Found in model_renderer.pose_renderer. This renderer required a blender complied as a python module. Models can be loaded and hidden for faster rendering, as loading the model tends to be the bottle neck. Only one instance of this class can exist per python interpreter, so it is not the best version for paralleization. 

```python
renderer = BpyRenderer()
renderer.loadModel('model.obj')
quats = [np.array([0,0,0,1]),] # list of unit quaternions (numpy array or size 4)
# Returns rendered images as numpy arrays
imgs = renderer.renderPose(quats)
# Save renders to disk
image_filenames = ['render_0.png',]
renderer.renderPose(quats, image_filenames)
```

Syscall Renderer
-----
Found in model_renderer.syscall_renderer. This rendere works with the standard blender install, as it simply makes system calls to the blender executable. Set the system variable BLENDER_PATH to the blender executable on your system. This renderer creates and loads the models everytime it is called, so it is not ideal for iterative rendering. It is, however, easy to parrelelize for batch rendering. See render_scripts/render_model_set.py.

```python
quats = [np.array([0,0,0,1]),]
# Returns rendered images as numpy arrays
imgs = renderView('model.obj', quats)
# Save renders to disk
image_filenames = ['render_0.png',]
renderView('model.obj', quats, image_filenames)
```

Batch Rendering
-----
To render all models in a dataset, use render_scripts/render_model_set.py. You can pass the models as single model path, txt file containing a list of models, or a director where all the models are saved. The folder structure the renderer expects in class/model/mesh.obj. The mesh_dir_depth should be equal to the depth of the subfolders between the model folder and the mesh.

```
python render_scripts/render_model_set.py --models /path/to/objs --render_folder /path/to/render/too --num_poses 500 --num_workers 32
```
