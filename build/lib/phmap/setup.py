from setuptools import setup, find_packages
from pathlib import Path

# Read version from version.py (same directory as setup.py)
version_file = Path(__file__).parent / 'version.py'
exec(open(version_file).read())

# Read README if exists (same directory as setup.py)
readme_file = Path(__file__).parent / 'README.md'
long_description = ''
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

# Since setup.py is inside the phmap package directory,
# we need to tell setuptools that the current directory IS the phmap package
# Find all subpackages and add 'phmap' as the root package
subpackages = find_packages(where='.')
packages = ['phmap'] + [f'phmap.{pkg}' for pkg in subpackages]

setup(
    name='phmap',
    version=__version__,
    description='PH-Map: Multi-task cell type classification package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='PH-Map Team',
    author_email='',
    url='',
    packages=packages,
    package_dir={
        'phmap': '.',  # Current directory is the phmap package root
    },
    package_data={
        'phmap': [
            'models/full_model/*.pth',
            'models/full_model/*.pkl',
        ],
    },
    include_package_data=True,
    install_requires=[
        'torch>=2.0.0',
        'scanpy>=1.9.0',
        'pandas>=1.5.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'anndata>=0.9.0',
        'scipy>=1.9.0',
    ],
    python_requires='>=3.11',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
    ],
    keywords='single-cell RNA-seq cell type classification multi-task learning',
)
