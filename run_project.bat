@echo off
python src\generate_data.py
python src\train.py
python -m streamlit run app.py
