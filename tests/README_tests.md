# Quick testing

## Unit tests
Install pytest (optional):
```powershell
pip install pytest
pytest -q
```

## Manual tests
1. Run server: `python app.py`
2. Open `/live` and confirm stream.
3. Open `/dashboard` and confirm new frames appear.
4. Click frame thumbnail: should open modal.
5. Generate PDF report.
6. Enable EMAIL and send to yourself.
