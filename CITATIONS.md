# Sitasi dan Sumber

Daftar ini dilengkapi seiring pengerjaan. Versi diambil dari lingkungan pengembangan (`pip freeze`).

## Library

| Library | Versi | Referensi |
|---|---|---|
| TensorFlow | 2.20.0 | Abadi, M. dkk. (2015). *TensorFlow: Large-scale machine learning on heterogeneous systems.* https://www.tensorflow.org |
| Keras | 3.10.0 | Chollet, F. dkk. (2015). *Keras.* https://keras.io |
| scikit-learn | 1.6.1 | Pedregosa, F. dkk. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. |
| pandas | 2.3.3 | McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56-61. |
| NumPy | 2.0.2 | Harris, C. R. dkk. (2020). Array programming with NumPy. *Nature*, 585, 357-362. |
| Matplotlib | 3.9.4 | Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90-95. |
| SQLAlchemy | 2.0.55 | Bayer, M. (2012). SQLAlchemy. Dalam *The Architecture of Open Source Applications, Vol. II.* https://www.sqlalchemy.org |
| joblib | 1.5.3 | https://joblib.readthedocs.io |

## Library tambahan (API, Generative AI, web, pengujian)

| Library / layanan | Referensi |
|---|---|
| FastAPI | Ramírez, S. *FastAPI.* https://fastapi.tiangolo.com |
| Uvicorn | https://www.uvicorn.org |
| Pydantic | https://docs.pydantic.dev |
| Google Gemini API dan SDK `google-genai` | https://ai.google.dev/gemini-api/docs |
| Streamlit | https://streamlit.io |
| Requests | https://requests.readthedocs.io |
| python-dotenv | https://github.com/theskumar/python-dotenv |
| pytest, httpx | https://docs.pytest.org , https://www.python-httpx.org |

Versi paket di atas diambil dari `pip freeze` (isi setelah semuanya terpasang di venv).
Bila RAG dikerjakan, tambahkan vector database dan model embedding yang dipilih.

## Data

- **Dataset pesan customer:** sintetis, dibuat sendiri dengan `src/generate_dataset.py` (template dan slot berbahasa Indonesia). Tidak memakai data asli Firaza.
- **Test set manual dan blind:** ditulis sendiri oleh penulis proyek.
- **Konteks bisnis:** brief Use Case 2 Firaza, Dicoding Bootcamp Batch 15; situs mitra firaza.com.

## Catatan kepatuhan

Model klasifikasi dilatih dari nol: tidak memakai TensorFlow Hub atau model pra-latih, tidak memakai AutoML, dan tidak berasal dari fork repositori AI milik pihak lain.