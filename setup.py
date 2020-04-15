from setuptools import setup

def readme():
    with open('README.MD') as f:
        return f.read()

setup(name='pyfocusr',
      version='0.0.1',
      description='Python Implementation of the FOCUSR Algorithm',
      long_description=readme(),
      url='',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
      ],
      keywords='graph, laplacian, spectral coordinates, point cloud, registration, mesh, surface',
      author='Anthony Gatti',
      author_email='anthony@neuralseg.com',
      license='MIT',
      packages=['pyfocusr'],
      install_requires=['numpy', 'future', 'scipy', 'cycpd', 'itkwidgets', 'matplotlib', 'vtk'],
      zip_safe=False)
