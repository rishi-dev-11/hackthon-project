from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()
    # Filter out comments and empty lines
    requirements = [req for req in requirements if req and not req.startswith("#")]
    # Remove any platform-specific markers for pip install
    requirements = [req.split(";")[0].strip() if ";" in req else req for req in requirements]

setup(
    name="documorph-ai",
    version="1.0.0",
    author="DocuMorph Team",
    author_email="yourname@example.com",
    description="Intelligent document transformation platform using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/documorph-ai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "documorph=backend.documorph_ai:main",
        ],
    },
) 