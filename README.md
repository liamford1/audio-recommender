# Audio-Based Music Recommender

An intelligent music recommendation system that analyzes the actual sound characteristics of songs to suggest similar tracks, rather than relying solely on metadata or collaborative filtering.

## Project Overview

This system extracts audio embeddings from songs using pre-trained deep learning models and recommends music based on acoustic similarity. The project demonstrates content-based recommendation systems, audio signal processing, and machine learning evaluation techniques.

## Features

- Audio-based similarity using deep learning embeddings
- Spotify API integration for live music data
- Comprehensive evaluation metrics (precision@k, diversity, novelty)
- Web interface for interactive recommendations
- Hybrid approach combining audio features with metadata

## Tech Stack

- **Audio Processing**: librosa, OpenL3
- **Machine Learning**: scikit-learn, numpy, pandas
- **APIs**: Spotify Web API
- **Web Framework**: Flask
- **Database**: SQLite

## Setup Instructions

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your API keys
6. Run setup scripts (see docs/SETUP.md)

## Project Status

🚧 **In Development** - Week 1: Foundation & Core ML

See PROGRESS.md for detailed development tracking.
