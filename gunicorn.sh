#!/bin/sh
gunicorn app:app -w 1 --threads 8 -b 0.0.0.0:5000
