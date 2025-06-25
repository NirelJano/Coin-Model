#!/bin/bash

# This script sets up the Python virtual environment for the project.
# To run it, make it executable first: `chmod +x setup.sh`, then run: `./setup.sh`

# הפעל סביבה וירטואלית אם קיימת, אחרת צור אחת
if [ ! -d "venv" ]; then
    /usr/local/bin/python3.10 -m venv venv
fi

source venv/bin/activate

# התקנת כל החבילות לפי requirements.txt
pip install --upgrade pip
pip install --upgrade -r requirements.txt

echo "הסביבה מוכנה! כדי להפעיל את הקוד:"
echo "source venv/bin/activate"
echo "python run_rfdetr.py"