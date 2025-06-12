# shipment_optimization_AWS
The shipment optimization model that is deployed on AWS, gets retrained through github push, and infers on new data online.










shipment_optimization_AWS/
│
├── src/
│   ├── prepare_data.py          # Reads from S3
│   ├── train_model.py           # Trains & saves model
│   └── optimize_conditions.py   # Used by Streamlit
│
├── app/
│   └── streamlit_app.py         # User-facing UI
│
├── models/
│   └── (stores model if local)
│
├── .github/
│   └── workflows/
│       └── train_on_push.yml    # GitHub Actions workflow
│
├── requirements.txt
└── README.md
