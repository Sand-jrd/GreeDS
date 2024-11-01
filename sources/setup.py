from setuptools import setup, find_packages
from pip._internal.req import parse_requirements


reqs = parse_requirements("../requirements.txt", session=False)
try:
    reqs = [str(ir.req) for ir in reqs]
except:
    reqs = [str(ir.requirement) for ir in reqs]

setup(
    version="2.0",
    name="GreeDS",  
    description='This package is a ADI or ARDI sequence processing tool that aim to distangle extended signal (like disks) from quasi-static speakels) using iterative PCA',
    url='https://github.com/Sand-jrd/GreeDS',
    author='Sandrine Juillard',
    author_email='sjuillard@uliege.be',
    packages=find_packages(),
    install_requires=reqs,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)