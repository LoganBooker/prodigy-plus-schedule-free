from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

setup(
  name = 'prodigy-plus-schedule-free',
  packages = find_packages(exclude=[]),
  version = '2.0.0-rc1',
  license='Apache 2.0',
  description = 'Automatic learning rate optimiser based on Prodigy and Schedule-Free',
  author = 'Logan Booker',
  author_email = 'me@loganbooker.dev',
  long_description=README,
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/LoganBooker/prodigy-plus-schedule-free',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers'
  ],
  install_requires=[
    'torch>=2.0'
  ],
  classifiers=[
    "Programming Language :: Python :: 3",
    'License :: OSI Approved :: Apache Software License',
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.4',
)
