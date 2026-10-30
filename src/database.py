"""
database.py - Skema database Firaza Custom Order Management (SQLite + SQLAlchemy 2.x).

Tabel:
  customers       data customer (nama, kontak, kanal)
  products        katalog produk (sajadah, handuk, hampers, packaging)
  vendors         vendor bordir/printing eksternal
  orders          satu pesanan: pesan asli, hasil klasifikasi AI, status, ringkasan & kartu ucapan
  order_items     rincian barang per pesanan (produk, jumlah, nama/teks custom, font, layout)
  status_history  riwayat perubahan status (audit trail + "riwayat custom order")

Alur status (hanya maju, kecuali revisi dari Production kembali ke Design/Preparation):
  New -> Design/Preparation -> Production -> Completed

Pakai:
  python src/database.py --db data/firaza.db --seed      # buat tabel + isi katalog contoh
  from database import init_db, change_status, ...
"""
import argparse
import enum
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text,
                        create_engine, event, select)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


def utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class OrderStatus(str, enum.Enum):
    NEW = "New"
    DESIGN = "Design/Preparation"
    PRODUCTION = "Production"
    COMPLETED = "Completed"


# Sama persis dengan LABELS di common.py (model klasifikasi)
ORDER_TYPES = ["Ready Stock", "Custom Sajadah", "Custom Handuk", "Hampers", "Other"]
CHANNELS = ["Shopee", "TikTok Shop", "Website", "WhatsApp", "Lainnya"]
PRODUCT_CATEGORIES = ["sajadah", "handuk", "hampers", "packaging", "lainnya"]
FONT_OPTIONS = ["Kaligrafi", "Script", "Arab", "Elegan", "Klasik", "Sans Simple", "Bold", "Handwriting"]
LAYOUT_OPTIONS = ["Tengah", "Atas", "Bawah", "Pojok Kanan Bawah", "Pojok Kiri Atas"]

# Transisi status yang diizinkan
ALLOWED_TRANSITIONS = {
    OrderStatus.NEW: {OrderStatus.DESIGN},
    OrderStatus.DESIGN: {OrderStatus.PRODUCTION},
    OrderStatus.PRODUCTION: {OrderStatus.COMPLETED, OrderStatus.DESIGN},  # DESIGN = revisi
    OrderStatus.COMPLETED: set(),
}


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    phone: Mapped[Optional[str]] = mapped_column(String(30))
    channel: Mapped[str] = mapped_column(String(30), default="WhatsApp")
    address: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    orders: Mapped[list["Order"]] = relationship(back_populates="customer")


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(String(40), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    category: Mapped[str] = mapped_column(String(30))
    is_customizable: Mapped[bool] = mapped_column(default=False)
    price_idr: Mapped[Optional[int]] = mapped_column(Integer)   # diisi sesuai katalog asli
    active: Mapped[bool] = mapped_column(default=True)


class Vendor(Base):
    __tablename__ = "vendors"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    service: Mapped[str] = mapped_column(String(30))         # bordir / printing
    contact: Mapped[Optional[str]] = mapped_column(String(60))
    notes: Mapped[Optional[str]] = mapped_column(Text)


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    order_code: Mapped[str] = mapped_column(String(20), unique=True)       # contoh: FRZ-0001
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    channel: Mapped[str] = mapped_column(String(30), default="WhatsApp")
    raw_message: Mapped[Optional[str]] = mapped_column(Text)                  # pesan asli customer
    predicted_type: Mapped[Optional[str]] = mapped_column(Enum(*ORDER_TYPES, name="order_type_pred"))
    predicted_confidence: Mapped[Optional[float]] = mapped_column(Float)
    order_type: Mapped[Optional[str]] = mapped_column(Enum(*ORDER_TYPES, name="order_type"))  # final (dikonfirmasi admin)
    status: Mapped[OrderStatus] = mapped_column(Enum(OrderStatus, values_callable=lambda e: [x.value for x in e]),
                                                default=OrderStatus.NEW)
    ai_summary: Mapped[Optional[dict]] = mapped_column(JSON)                  # structured order summary (GenAI)
    greeting_card: Mapped[Optional[str]] = mapped_column(Text)                # draft kartu ucapan (GenAI)
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    customer: Mapped[Customer] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(back_populates="order", cascade="all, delete-orphan")
    history: Mapped[list["StatusHistory"]] = relationship(back_populates="order", cascade="all, delete-orphan",
                                                          order_by="StatusHistory.id")


class OrderItem(Base):
    __tablename__ = "order_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    product_id: Mapped[Optional[int]] = mapped_column(ForeignKey("products.id"))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    color: Mapped[Optional[str]] = mapped_column(String(40))
    custom_text: Mapped[Optional[str]] = mapped_column(String(200))           # nama/teks yang dicetak atau dibordir
    font: Mapped[Optional[str]] = mapped_column(String(40))
    layout: Mapped[Optional[str]] = mapped_column(String(40))
    vendor_id: Mapped[Optional[int]] = mapped_column(ForeignKey("vendors.id"))
    item_notes: Mapped[Optional[str]] = mapped_column(Text)
    order: Mapped[Order] = relationship(back_populates="items")
    product: Mapped[Optional[Product]] = relationship()
    vendor: Mapped[Optional[Vendor]] = relationship()


class StatusHistory(Base):
    __tablename__ = "status_history"
    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    from_status: Mapped[Optional[str]] = mapped_column(String(30))
    to_status: Mapped[str] = mapped_column(String(30))
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    note: Mapped[Optional[str]] = mapped_column(Text)
    order: Mapped[Order] = relationship(back_populates="history")


# ---------------------------------------------------------------- helper
def get_engine(db_path="data/firaza.db"):
    engine = create_engine(f"sqlite:///{db_path}")

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _):          # SQLite mengabaikan FOREIGN KEY kecuali diaktifkan
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def init_db(db_path="data/firaza.db"):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def next_order_code(session):
    n = session.scalar(select(Order.id).order_by(Order.id.desc()).limit(1)) or 0
    return f"FRZ-{n + 1:04d}"


def create_order(session, customer, raw_message=None, channel="WhatsApp", **fields):
    """Buat order baru berstatus New + catat entri pertama di status_history."""
    order = Order(order_code=next_order_code(session), customer=customer, raw_message=raw_message,
                  channel=channel, status=OrderStatus.NEW, **fields)
    order.history.append(StatusHistory(from_status=None, to_status=OrderStatus.NEW.value, note="Order dibuat"))
    session.add(order)
    session.flush()
    return order


def change_status(session, order, new_status, note=None):
    """Ubah status sesuai ALLOWED_TRANSITIONS dan catat riwayat. Raise ValueError bila tidak valid."""
    new_status = OrderStatus(new_status)
    if new_status not in ALLOWED_TRANSITIONS[order.status]:
        raise ValueError(f"Transisi tidak valid: {order.status.value} -> {new_status.value}")
    order.history.append(StatusHistory(from_status=order.status.value, to_status=new_status.value, note=note))
    order.status = new_status
    session.flush()
    return order


SEED_PRODUCTS = [
    ("SJD-POL", "Sajadah Polos", "sajadah", True),
    ("SJD-TRV", "Sajadah Travel", "sajadah", True),
    ("HDK-MND", "Handuk Mandi", "handuk", True),
    ("HDK-MKA", "Handuk Muka", "handuk", True),
    ("HDK-TGN", "Handuk Tangan", "handuk", True),
    ("HMP-HS", "Paket Hampers Handuk + Sajadah", "hampers", True),
    ("PKG-BOX", "Box Packaging", "packaging", False),
    ("PKG-KRT", "Kartu Ucapan", "packaging", False),
]


def seed(engine):
    """Isi katalog contoh. Harga sengaja dikosongkan; isi sesuai katalog Firaza yang sebenarnya."""
    with Session(engine) as s:
        if s.scalar(select(Product.id).limit(1)) is None:
            s.add_all([Product(sku=a, name=b, category=c, is_customizable=d) for a, b, c, d in SEED_PRODUCTS])
            s.commit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/firaza.db")
    ap.add_argument("--seed", action="store_true")
    a = ap.parse_args()
    eng = init_db(a.db)
    if a.seed:
        seed(eng)
    print(f"Database siap: {a.db}")