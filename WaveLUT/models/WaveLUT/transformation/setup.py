import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 设置 CUDA 环境变量
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.1'
os.environ['PATH'] = '/usr/local/cuda-12.1/bin:' + os.environ['PATH']

def get_version(version_file):
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

# 获取源代码路径
os.chdir(osp.dirname(osp.abspath(__file__)))
csrc_directory = osp.join('WaveLUT', 'csrc')

setup(
    name='WaveLUT',
    version=get_version(osp.join('WaveLUT', 'version.py')),
    packages=find_packages(),
    include_package_data=False,
    ext_modules=[
        CUDAExtension(
            name='WaveLUT._ext',
            sources=[
                osp.join(csrc_directory, 'transformation.cpp'),
                osp.join(csrc_directory, 'transformation_cpu.cpp'),
                osp.join(csrc_directory, 'transformation_cuda.cu')
            ],
            # 添加额外的编译选项
            extra_compile_args={
                'cxx': ['-std=c++14'],  # C++ 编译选项
                'nvcc': [
                    '-std=c++14',
                    '--expt-relaxed-constexpr',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__'
                ],  # CUDA 编译选项
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    license='Apache License 2.0',
    zip_safe=False
)
