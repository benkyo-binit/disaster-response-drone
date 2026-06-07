# Surveillance and Disaster Response Drone

A disaster response platform that combines drone-based data collection, real-time video streaming, AI analysis, and a custom Ground Control Station for remote monitoring and emergency response.

## Overview

This project aims to assist disaster response operations by providing live aerial video, GPS tracking, and AI-powered scene analysis. A Raspberry Pi onboard the drone captures and streams data over a 4G network to a Ground Control Station, where information is processed and visualized for operators.

## Features

* Real-time video streaming over 4G networks
* Custom Ground Control Station (GCS)
* GPS visualization and tracking
* Geotagged event logging
* Dataset collection for AI training
* Disaster scene classification
* Modular software architecture

## System Components

### Drone System

* Raspberry Pi
* Camera Module
* GPS Module
* 4G Connectivity

### Ground Control Station

* Live video monitoring
* GPS map display
* Event logging dashboard
* AI analysis interface

## Technologies Used

* Python
* Raspberry Pi
* PyTorch
* OpenCV
* RTMP Streaming
* GPS Integration

## Workflow

Drone Camera
↓
Raspberry Pi
↓
4G Network
↓
Ground Control Station
├── Live Video Feed
├── GPS Tracking
├── Event Logging
└── AI Analysis

## Current Development

* Ground Control Station development
* AI model training and evaluation
* Sensor integration
* Streaming optimization

