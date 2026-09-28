python src/generate_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python src/train.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m streamlit run app.py
