"""Setup script for the Agentor package.

This setup script is primarily for backward compatibility with tools that don't
support pyproject.toml yet. The primary build configuration is in pyproject.toml.
"""

import re
from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Extract version from pyproject.toml to ensure consistency
def get_version():
    with open("pyproject.toml", encoding="utf-8") as f:
        content = f.read()
    version_match = re.search(r'version = "([^"]+)"', content)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version in pyproject.toml")

setup(
    name="agentor",
    version=get_version(),
    description="A flexible agent framework with asynchronous support, enhanced error handling, caching, security, and monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Agentor Team",
    author_email="info@agentor.ai",
    url="https://github.com/agentor-ai/agentor",
    project_urls={
        "Homepage": "https://agentor.ai",
        "Documentation": "https://docs.agentor.ai",
        "Source": "https://github.com/agentor-ai/agentor",
        "Issues": "https://github.com/agentor-ai/agentor/issues",
    },
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    keywords="agent, framework, async, llm, ai, machine-learning",
    extras_require={
        "pinecone": ["pinecone-client>=2.2.1"],
        "qdrant": ["qdrant-client>=1.1.1"],
        "weaviate": ["weaviate-client>=3.15.4"],
        "milvus": ["milvus>=2.2.8"],
        "all-vector-dbs": [
            "pinecone-client>=2.2.1",
            "qdrant-client>=1.1.1",
            "weaviate-client>=3.15.4",
            "milvus>=2.2.8"
        ],
        "dev": [
            "pytest>=7.3.1",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.24.1",
            "flake8>=6.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.5.1"
        ],
    },
)
