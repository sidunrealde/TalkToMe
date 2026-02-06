from setuptools import setup, find_packages

setup(
    name="talktome",
    version="0.1.0",
    description="AI Avatar System with Pipecat and Ditto TalkingHead",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pipecat-ai[groq,silero]>=0.0.102",
        "smallwebrtc>=0.1.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "python-multipart>=0.0.6",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "onnxruntime-gpu>=1.16.0",
        "safetensors>=0.4.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "scipy>=1.11.0",
        "httpx>=0.26.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "talktome=src.server:main",
        ],
    },
)
