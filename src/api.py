"""
api.py - REST API + Inference API Firaza Custom Order Management (FastAPI).

Jalankan dari ROOT proyek:
    uvicorn api:app --app-dir src --reload
Dokumentasi interaktif (Swagger): http://127.0.0.1:8000/docs   |   ReDoc: /redoc

Konfigurasi lewat environment / .env:
    FIRAZA_DB     path database SQLite   (default: data/firaza.db)
    MODEL_DIR     folder model beku      (default: models/final)
    GEMINI_API_KEY, GEMINI_MODEL         (lihat genai_service.py; tanpa key, fitur GenAI memakai fallback)
    CORS_ORIGINS  daftar origin dipisah koma (default: *, hanya untuk pengembangan)
"""
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import schemas as S
from database import (ALLOWED_TRANSITIONS, CHANNELS, FONT_OPTIONS, LAYOUT_OPTIONS, ORDER_TYPES, Customer, Order,
                      OrderItem, OrderStatus, Product, Vendor, change_status, create_order, get_engine, init_db, seed)
from genai_service import GenAIService, extract_basic


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = os.getenv("FIRAZA_DB", "data/firaza.db")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    app.state.engine = init_db(db_path)
    seed(app.state.engine)
    app.state.db_path = db_path
    app.state.genai = GenAIService()
    app.state.classifier, app.state.classifier_error = None, None
    try:
        from predictor import OrderClassifier

        app.state.classifier = OrderClassifier(os.getenv("MODEL_DIR", "models/final"))
    except Exception as e:  # model belum dilatih / path salah: API tetap hidup, /classify memberi 503
        app.state.classifier_error = f"{type(e).__name__}: {e}"
    yield
    app.state.engine.dispose()


app = FastAPI(
    title="Firaza Custom Order API",
    version="0.1.0",
    description=("API manajemen custom order Firaza: klasifikasi jenis order (model sendiri: ensemble CNN + LogReg), "
                 "CRUD customer/produk/order, alur status New → Design/Preparation → Production → Completed, "
                 "serta fitur Generative AI (structured order summary dan draft kartu ucapan)."),
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


def get_db(request: Request):
    with Session(request.app.state.engine) as session:
        yield session


def get_or_404(db: Session, model, obj_id: int, label: str):
    obj = db.get(model, obj_id)
    if obj is None:
        raise HTTPException(404, f"{label} dengan id {obj_id} tidak ditemukan")
    return obj


def classifier_or_503(request: Request):
    clf = request.app.state.classifier
    if clf is None:
        raise HTTPException(503, f"Model klasifikasi belum dimuat ({request.app.state.classifier_error}). "
                                 "Jalankan: python src/train_final.py")
    return clf


def check_refs(db: Session, product_id: Optional[int], vendor_id: Optional[int]):
    if product_id is not None and db.get(Product, product_id) is None:
        raise HTTPException(422, f"product_id {product_id} tidak ditemukan")
    if vendor_id is not None and db.get(Vendor, vendor_id) is None:
        raise HTTPException(422, f"vendor_id {vendor_id} tidak ditemukan")


def items_for_genai(order: Order) -> List[Dict[str, Any]]:
    return [{"produk": it.product.name if it.product else None, "jumlah": it.quantity, "warna": it.color,
             "teks_custom": it.custom_text, "font": it.font, "layout": it.layout} for it in order.items]


# ================================================================== umum
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/health", tags=["Umum"])
def health(request: Request):
    st = request.app.state
    return {"status": "ok", "database": st.db_path, "classifier_loaded": st.classifier is not None,
            "classifier_error": st.classifier_error, "gemini_configured": st.genai.available, "gemini_model": st.genai.model}


@app.get("/options", tags=["Umum"], summary="Pilihan tetap untuk dropdown di web")
def options():
    return {"order_types": ORDER_TYPES, "statuses": [s.value for s in OrderStatus], "channels": CHANNELS,
            "fonts": FONT_OPTIONS, "layouts": LAYOUT_OPTIONS,
            "status_transitions": {k.value: sorted(v.value for v in vs) for k, vs in ALLOWED_TRANSITIONS.items()}}


# ================================================================== inference API
@app.post("/classify", response_model=S.ClassifyResponse, tags=["Inference"],
          summary="Klasifikasikan pesan customer ke 5 jenis order")
def classify(payload: S.ClassifyRequest, request: Request):
    return classifier_or_503(request).predict(payload.text)


# ================================================================== customers
@app.post("/customers", response_model=S.CustomerOut, status_code=201, tags=["Customers"])
def create_customer(payload: S.CustomerCreate, db: Session = Depends(get_db)):
    c = Customer(**payload.model_dump())
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


@app.get("/customers", response_model=List[S.CustomerOut], tags=["Customers"])
def list_customers(q: Optional[str] = Query(None, description="Cari nama atau telepon"),
                   limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    stmt = select(Customer).order_by(Customer.id.desc())
    if q:
        stmt = stmt.where(or_(Customer.name.ilike(f"%{q}%"), Customer.phone.ilike(f"%{q}%")))
    return db.scalars(stmt.limit(limit).offset(offset)).all()


@app.get("/customers/{customer_id}", response_model=S.CustomerOut, tags=["Customers"])
def get_customer(customer_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Customer, customer_id, "Customer")


@app.put("/customers/{customer_id}", response_model=S.CustomerOut, tags=["Customers"])
def update_customer(customer_id: int, payload: S.CustomerUpdate, db: Session = Depends(get_db)):
    c = get_or_404(db, Customer, customer_id, "Customer")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(c, k, v)
    db.commit()
    db.refresh(c)
    return c


@app.delete("/customers/{customer_id}", status_code=204, tags=["Customers"])
def delete_customer(customer_id: int, db: Session = Depends(get_db)):
    c = get_or_404(db, Customer, customer_id, "Customer")
    if c.orders:
        raise HTTPException(409, f"Customer masih punya {len(c.orders)} order; hapus atau pindahkan order dulu")
    db.delete(c)
    db.commit()
    return Response(status_code=204)


# ================================================================== products
@app.get("/products", response_model=List[S.ProductOut], tags=["Products"])
def list_products(category: Optional[str] = None, active_only: bool = True, db: Session = Depends(get_db)):
    stmt = select(Product).order_by(Product.id)
    if category:
        stmt = stmt.where(Product.category == category)
    if active_only:
        stmt = stmt.where(Product.active.is_(True))
    return db.scalars(stmt).all()


@app.get("/products/{product_id}", response_model=S.ProductOut, tags=["Products"])
def get_product(product_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Product, product_id, "Produk")


@app.post("/products", response_model=S.ProductOut, status_code=201, tags=["Products"])
def create_product(payload: S.ProductCreate, db: Session = Depends(get_db)):
    p = Product(**payload.model_dump())
    db.add(p)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, f"SKU '{payload.sku}' sudah dipakai")
    db.refresh(p)
    return p


@app.put("/products/{product_id}", response_model=S.ProductOut, tags=["Products"])
def update_product(product_id: int, payload: S.ProductUpdate, db: Session = Depends(get_db)):
    p = get_or_404(db, Product, product_id, "Produk")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(p, k, v)
    db.commit()
    db.refresh(p)
    return p


# ================================================================== orders
@app.post("/orders", response_model=S.OrderOut, status_code=201, tags=["Orders"],
          summary="Buat order baru (opsional: klasifikasi otomatis dari pesan customer)")
def create_order_endpoint(payload: S.OrderCreate, request: Request, db: Session = Depends(get_db)):
    customer = get_or_404(db, Customer, payload.customer_id, "Customer")
    for it in payload.items:
        check_refs(db, it.product_id, it.vendor_id)
    fields: Dict[str, Any] = {"due_date": payload.due_date, "notes": payload.notes}
    pred = None
    clf = request.app.state.classifier
    if payload.auto_classify and payload.raw_message and clf is not None:
        pred = clf.predict(payload.raw_message)
        fields.update(predicted_type=pred["label"], predicted_confidence=pred["confidence"])
    order_type = payload.order_type or (pred["label"] if (payload.confirm_prediction and pred) else None)
    order = create_order(db, customer, raw_message=payload.raw_message, channel=payload.channel,
                         order_type=order_type, **fields)
    for it in payload.items:
        order.items.append(OrderItem(**it.model_dump()))
    db.commit()
    db.refresh(order)
    return order


@app.get("/orders", response_model=List[S.OrderOut], tags=["Orders"])
def list_orders(status: Optional[S.StatusLit] = None, order_type: Optional[S.OrderTypeLit] = None,
                customer_id: Optional[int] = None, q: Optional[str] = Query(None, description="Cari kode order atau isi pesan"),
                limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    stmt = select(Order).order_by(Order.id.desc())
    if status:
        stmt = stmt.where(Order.status == OrderStatus(status))
    if order_type:
        stmt = stmt.where(Order.order_type == order_type)
    if customer_id:
        stmt = stmt.where(Order.customer_id == customer_id)
    if q:
        stmt = stmt.where(or_(Order.order_code.ilike(f"%{q}%"), Order.raw_message.ilike(f"%{q}%")))
    return db.scalars(stmt.limit(limit).offset(offset)).all()


@app.get("/orders/{order_id}", response_model=S.OrderOut, tags=["Orders"])
def get_order(order_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Order, order_id, "Order")


@app.put("/orders/{order_id}", response_model=S.OrderOut, tags=["Orders"],
         summary="Ubah data order (mis. konfirmasi jenis order final)")
def update_order(order_id: int, payload: S.OrderUpdate, db: Session = Depends(get_db)):
    order = get_or_404(db, Order, order_id, "Order")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(order, k, v)
    db.commit()
    db.refresh(order)
    return order


@app.delete("/orders/{order_id}", status_code=204, tags=["Orders"])
def delete_order(order_id: int, db: Session = Depends(get_db)):
    db.delete(get_or_404(db, Order, order_id, "Order"))
    db.commit()
    return Response(status_code=204)


@app.post("/orders/{order_id}/status", response_model=S.OrderOut, tags=["Orders"],
          summary="Ubah status order (hanya transisi yang diizinkan)")
def change_order_status(order_id: int, payload: S.StatusChange, db: Session = Depends(get_db)):
    order = get_or_404(db, Order, order_id, "Order")
    try:
        change_status(db, order, payload.status, payload.note)
    except ValueError as e:
        allowed = sorted(s.value for s in ALLOWED_TRANSITIONS[order.status])
        raise HTTPException(409, detail={"message": str(e), "status_saat_ini": order.status.value, "transisi_diizinkan": allowed})
    db.commit()
    db.refresh(order)
    return order


@app.get("/orders/{order_id}/history", response_model=List[S.StatusHistoryOut], tags=["Orders"],
         summary="Riwayat perubahan status")
def order_history(order_id: int, db: Session = Depends(get_db)):
    return get_or_404(db, Order, order_id, "Order").history


# ---- item order
@app.post("/orders/{order_id}/items", response_model=S.OrderItemOut, status_code=201, tags=["Order items"])
def add_item(order_id: int, payload: S.OrderItemIn, db: Session = Depends(get_db)):
    order = get_or_404(db, Order, order_id, "Order")
    check_refs(db, payload.product_id, payload.vendor_id)
    item = OrderItem(**payload.model_dump())
    order.items.append(item)
    db.commit()
    db.refresh(item)
    return item


@app.put("/orders/{order_id}/items/{item_id}", response_model=S.OrderItemOut, tags=["Order items"])
def update_item(order_id: int, item_id: int, payload: S.OrderItemUpdate, db: Session = Depends(get_db)):
    item = db.get(OrderItem, item_id)
    if item is None or item.order_id != order_id:
        raise HTTPException(404, f"Item {item_id} tidak ditemukan pada order {order_id}")
    data = payload.model_dump(exclude_unset=True)
    check_refs(db, data.get("product_id"), data.get("vendor_id"))
    for k, v in data.items():
        setattr(item, k, v)
    db.commit()
    db.refresh(item)
    return item


@app.delete("/orders/{order_id}/items/{item_id}", status_code=204, tags=["Order items"])
def delete_item(order_id: int, item_id: int, db: Session = Depends(get_db)):
    item = db.get(OrderItem, item_id)
    if item is None or item.order_id != order_id:
        raise HTTPException(404, f"Item {item_id} tidak ditemukan pada order {order_id}")
    db.delete(item)
    db.commit()
    return Response(status_code=204)


# ================================================================== Generative AI
@app.post("/orders/{order_id}/summary", tags=["Generative AI"],
          summary="Structured order summary (Gemini; fallback berbasis aturan bila tidak tersedia)")
def order_summary(order_id: int, request: Request, force: bool = Query(False, description="Buat ulang walau sudah ada"),
                  db: Session = Depends(get_db)) -> Dict[str, Any]:
    order = get_or_404(db, Order, order_id, "Order")
    if order.ai_summary and not force:
        return order.ai_summary
    jenis = order.order_type or order.predicted_type
    summary = request.app.state.genai.summarize(order.raw_message, items_for_genai(order), jenis, use_cache=not force)
    summary["jenis_order"] = jenis
    order.ai_summary = summary
    db.commit()
    return summary


@app.post("/orders/{order_id}/greeting", response_model=S.GreetingOut, tags=["Generative AI"],
          summary="Draft kartu ucapan (Gemini; template bila tidak tersedia)")
def order_greeting(order_id: int, request: Request, payload: Optional[S.GreetingRequest] = None,
                   force: bool = Query(False, description="Buat draf baru (abaikan cache)"), db: Session = Depends(get_db)):
    order = get_or_404(db, Order, order_id, "Order")
    tone = payload.tone if payload else "hangat"
    names = ", ".join(dict.fromkeys(it.custom_text for it in order.items if it.custom_text))
    if not names:
        names = extract_basic(order.raw_message or "")["teks_custom"] or ""
    acara = (order.ai_summary or {}).get("acara")
    out = request.app.state.genai.greeting(order.raw_message, acara, names, tone, use_cache=not force)
    order.greeting_card = out["kartu_ucapan"]
    db.commit()
    return out