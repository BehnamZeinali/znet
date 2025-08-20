from setuptools import setup, find_packages

setup(
    name='zenet',
    version='0.1.0',
    packages=find_packages(),
    description='Zenet: A minimalist PyTorch-like deep learning library',
    author='Your Name',
    author_email='you@example.com',
    python_requires='>=3.7',
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
