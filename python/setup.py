from setuptools import setup, find_packages

setup(
    name="word_similarity_rs",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "word_similarity_rs": ["word_similarity_rs.so"]  # Include Rust binary
    },
    install_requires=[],
    author="Your Name",
    description="A Rust-based word pair similarity predictor wrapped as a Python library.",
    license="MIT",
    python_requires=">=3.8"
)