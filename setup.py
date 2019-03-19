from pathlib import Path
from setuptools import setup, find_packages

README = (Path(__file__).parent / "README.md").read_text()

REQUIRED = [
        "scikit-learn",
        "pandas",
        "numpy"
]

setup(
    name='lambdata_mac',
    version='0.1',
    description="Helper Functions",
    long_description=README,
    long_description_content_type="text/markdown",
    url = "https://github.com/macscheffer/lambdata_mac",
    packages=find_packages(),
    author="Mac Scheffer",
    install_requires=REQUIRED,
)
