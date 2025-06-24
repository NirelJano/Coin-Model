#!/bin/bash

# הפעל סביבה וירטואלית אם קיימת, אחרת צור אחת
if [ ! -d "venv" ]; then
    /usr/local/bin/python3.10 -m venv venv
fi

source venv/bin/activate

# התקנת כל החבילות לפי requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

echo "הסביבה מוכנה! כדי להפעיל את הקוד:"
echo "source venv/bin/activate"
echo "python run_rfdetr.py"