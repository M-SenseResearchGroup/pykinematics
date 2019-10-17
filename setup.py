import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# these lines allow 1 file to control the version, so only 1 file needs to be updated per version change
fid = open('pykinematics/version.py')
vers = fid.readlines()[-1].split()[-1].strip("\"'")
fid.close()

setuptools.setup(
    name="test",
    version=vers,
    author="Sample Author",
    author_email="author.email@email",
    description="Sample project description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/",  # project url, most likely a github link
    download_url="https://pypi.org/test",  # link to where the package can be downloaded, most likely PyPI
    project_urls={
        "Documentation": "https://test.readthedocs.io/en/latest/"
    },
    include_pacakge_data=False,  # set to True if you have data to package, ie models or similar
    # package_data={'package': ['*.csv']},  # currently adds any csv files alongside the top level __init__.py
    # package_data={'package.module': ['data.csv']},  # if data.csv is in a separate module
    packages=setuptools.find_packages(),  # automatically find required packages
    license='MIT',
    python_requires='>=3.6',  # Version of python required
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.7"
    ],
)
