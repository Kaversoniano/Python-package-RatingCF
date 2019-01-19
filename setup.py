import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RatingCF",
    version="0.1.2",
    author="Tianjian Yang",
    author_email="kaversoniano@gmail.com",
    description="A user-preference-sensitive collaborative filtering algorithm for recommender system, which is based on items' ratings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kaversoniano/Python-package-RatingCF",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
)