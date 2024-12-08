from setuptools import setup, find_packages

setup(
    name='UltraNovativAIBot',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'groq',
        'python-dotenv',
        'discord.py',
    ],
    entry_points={
        'console_scripts': [
            'ultranovativ-bot=src.main:main',
        ],
    },
)
