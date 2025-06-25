from setuptools import setup, find_packages

setup(
    name="ursa-minor",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.5.1",
        "transformers>=4.46.1",
        "trl>=0.12.0",
        "datasets>=3.1.0",
        "accelerate>=1.0.1",
        "vllm>=0.6.4.post1",
        "wandb>=0.17.3",
        "flash-attn>=2.0.0",
    ],
)