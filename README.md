# Plant Health, Nutrient & Post-Mortem Diagnostics

Upload a plant or leaf photo to get a health / N-P-K deficiency / dead-necrotic
prediction with confidence meters and agronomy recommendations.

Built with PyTorch (EfficientNet-B0 fine-tuning) and Streamlit.
Decision-support only: confirm with a soil or tissue test.

An optional Google Gemini explanation is available after a diagnosis. Set
`GEMINI_API_KEY` in your environment or enter it in the app sidebar. The image
is not sent to Gemini; only the classifier result and existing recommendations
are sent when you click **Generate AI explanation**.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Requires the trained checkpoint at `models/plant_health_effnet_b0.pt`.
