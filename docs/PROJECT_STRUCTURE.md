# Project Structure

```text
TruthLenseAi/
|-- backend/                 FastAPI API, model code, tests, and training pipeline
|-- docs/                    Project documentation and reference material
|-- frontend/                Browser-side configuration and JavaScript
|   |-- config.js            Supabase URL/key and API URL
|   `-- js/
|       |-- app.js           Homepage interactions and media analysis flow
|       `-- login.js         Email/password and OAuth sign-in flows
|-- supabase/
|   `-- migrations/          Database schema migrations
|-- AI_Deepfake_Misinformation_Synopsis_final.pdf  Existing reference PDF
|-- index.html               Homepage entry point
|-- login.html               Login entry point
`-- README.md                Project overview and setup notes
```

The HTML files stay at the repository root because their URLs are used by upload redirects, OAuth callbacks, and links within the app.
